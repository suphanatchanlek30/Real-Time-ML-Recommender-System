// Load test of GET /recommend.   make loadtest-smoke | loadtest | loadtest-stress
//
//   smoke   1 VU for 30 s: the system answers correctly
//   load    ramp to 100 VUs and hold: the target traffic. Thresholds: p95 < 100 ms, errors < 1%
//   stress  ramp the *request rate* until the service breaks, to find the bottleneck
//
// User ids come from users.json (make loadtest-users): a mix of users known to ALS, users with
// history in Redis and brand-new users, like the real traffic.
import http from 'k6/http';
import { check, sleep } from 'k6';
import { SharedArray } from 'k6/data';
import { Counter } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://api:8000';
const SCENARIO = __ENV.SCENARIO || 'load';
const users = new SharedArray('users', () => JSON.parse(open('./users.json')));
const wrongSource = new Counter('wrong_answers');  // 200 but not a valid recommendation

const scenarios = {
  smoke: {
    executor: 'constant-vus', vus: 1, duration: '30s',
  },
  load: {
    executor: 'ramping-vus', startVUs: 0, gracefulRampDown: '10s',
    stages: [
      { duration: '30s', target: 20 },
      { duration: '30s', target: 100 },
      { duration: '2m', target: 100 },  // hold
      { duration: '20s', target: 0 },
    ],
  },
  stress: {
    // open model: arrivals keep coming even when the service slows down, so overload shows up
    executor: 'ramping-arrival-rate', startRate: 100, timeUnit: '1s',
    preAllocatedVUs: 200, maxVUs: 2000,
    stages: [
      { duration: '30s', target: 300 },
      { duration: '30s', target: 600 },
      { duration: '30s', target: 1000 },
      { duration: '30s', target: 1500 },
      { duration: '30s', target: 2000 },
      { duration: '30s', target: 3000 },
      { duration: '20s', target: 0 },
    ],
  },
};

const thresholds = {
  http_req_failed: ['rate<0.01'],
  wrong_answers: ['count==0'],
};
if (SCENARIO !== 'stress') {
  thresholds.http_req_duration = ['p(95)<100'];  // the stress run is expected to break this
}

export const options = {
  scenarios: { [SCENARIO]: scenarios[SCENARIO] },
  thresholds,
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
  tags: { scenario: SCENARIO },
};

export default function () {
  const uid = users[Math.floor(Math.random() * users.length)];
  const res = http.get(`${BASE_URL}/recommend?user_id=${uid}&k=10`);
  const ok = check(res, {
    'status 200': (r) => r.status === 200,
    'ten items': (r) => r.status === 200 && r.json('items').length === 10,
    'ranked by the model': (r) => r.status === 200 && r.json('source') === 'ranker',
  });
  if (res.status === 200 && !ok) wrongSource.add(1);
  if (SCENARIO !== 'stress') sleep(0.1);  // think time of a real client
}

// keep the numbers: results/<scenario>.json is what the report is built from
export function handleSummary(data) {
  return {
    stdout: textSummary(data),
    [`/results/${SCENARIO}.json`]: JSON.stringify(data, null, 2),
  };
}

function textSummary(data) {
  const m = data.metrics;
  const t = m.http_req_duration.values;
  const line = (k, v) => `  ${k.padEnd(28)}${v}\n`;
  let out = `\n=== ${SCENARIO} ===\n`;
  out += line('requests', `${m.http_reqs.values.count}  (${m.http_reqs.values.rate.toFixed(0)}/s)`);
  out += line('failed', `${(m.http_req_failed.values.rate * 100).toFixed(2)} %`);
  out += line('latency avg / med', `${t.avg.toFixed(1)} / ${t.med.toFixed(1)} ms`);
  out += line('latency p90 / p95 / p99', `${t['p(90)'].toFixed(1)} / ${t['p(95)'].toFixed(1)} / ${t['p(99)'].toFixed(1)} ms`);
  out += line('latency max', `${t.max.toFixed(1)} ms`);
  if (m.vus_max) out += line('max VUs', `${m.vus_max.values.max}`);
  if (m.dropped_iterations) out += line('dropped iterations', `${m.dropped_iterations.values.count}`);
  for (const [name, th] of Object.entries(data.metrics).filter(([, v]) => v.thresholds)) {
    for (const [expr, res] of Object.entries(th.thresholds)) out += line(`${res.ok ? 'PASS' : 'FAIL'} ${name}`, expr);
  }
  return out + '\n';
}
