import { defineConfig } from 'vite';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

// Self-signed cert so a phone on the LAN gets a secure context, which
// getUserMedia requires on any origin that is not localhost. Generate with
// `openssl req -x509 ...` into .certs/; the phone shows a warning once and
// accepting it is expected.
const certDir = fileURLToPath(new URL('./.certs/', import.meta.url));
const https =
  fs.existsSync(certDir + 'cert.pem')
    ? {
        key: fs.readFileSync(certDir + 'key.pem'),
        cert: fs.readFileSync(certDir + 'cert.pem'),
      }
    : undefined;

export default defineConfig({
  // GitHub Pages serves this from /<repo>/, not from the domain root. Vite
  // bakes the base into asset URLs at build time, so it has to be set here
  // rather than fixed up afterwards. Overridable for other hosts.
  base: process.env.DEMO_BASE ?? '/webslam/',
  server: {
    // A phone cannot reach `localhost`; the demo is meant to be opened on a
    // device over the LAN (spec.md §8).
    host: true,
    https,
  },
  build: {
    target: 'es2022',
  },
});
