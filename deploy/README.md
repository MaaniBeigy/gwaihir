# gamebus-coach production deploy

Bootstrap once on the vGPU host:

```sh
docker volume create coach-chroma
docker volume create coach-models
```

Then place certs:

- `deploy/certs/server.crt`, `server.key`, `client-ca.pem` — for the nginx
  mTLS terminator (server cert presented to clients; CA bundle that signed
  the gateway client cert).
- `deploy/coach-certs/client.crt`, `client.key`, `gateway-server-ca.pem` —
  for the coach -> gateway webhook client.

Both directories are gitignored; nothing under them is committed.

Bring the stack up:

```sh
docker compose -f deploy/docker-compose.yml up -d
```
