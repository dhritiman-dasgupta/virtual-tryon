# Virtual try-on — production architecture

One 24 GB RTX 4090 runs two workloads that cannot be resident at the same
time. Everything below follows from that single constraint.

## Phases

    BROCHURE     Qwen3-VL-32B, 4-bit, ~20 GB, alone on the card.
                 Reads each garment photo once and writes a spec: type, piece
                 list, drape, colours, fabric. Cached to disk by image hash +
                 question hash + model id, so it is paid once per garment ever.

    GENERATE     ComfyUI (16.6 GB) + Qwen3-VL-4B guardrail (5.5 GB) = 22.1 GB.
                 Generates try-ons and inspects each one, reseeding on a
                 critical failure.

    IDLE         nothing resident.

Only one phase holds the GPU. The API owns the transition and refuses work
that belongs to the other phase rather than thrashing VRAM - an earlier build
that let a VLM and ComfyUI fight over memory turned an 8 s generation into
215 s, which is the failure this design exists to prevent.

## Endpoints

| method | path | phase | purpose |
|---|---|---|---|
| GET  | /healthz                  | any      | liveness |
| GET  | /readyz                   | any      | ComfyUI reachable, models present |
| GET  | /v1/phase                 | any      | which phase is loaded, VRAM in use |
| POST | /v1/phase                 | any      | switch phase (brochure/generate/idle) |
| POST | /v1/brochure              | brochure | analyse a set of garments |
| GET  | /v1/brochure/{id}         | any      | job status |
| GET  | /v1/brochure/{id}/specs   | any      | the finished specs |
| GET  | /v1/garments              | any      | every cached spec |
| POST | /v1/tryon                 | generate | generate, optionally guarded |
| GET  | /v1/jobs/{id}             | any      | job status, per-attempt detail |
| GET  | /v1/jobs/{id}/image       | any      | the result |
| POST | /v1/auth/token            | any      | exchange credentials for a token |

## Guardrail switch

Three levels, most specific wins:

    request   {"guardrail": false}      per call
    setting   GUARDRAIL_ENABLED=false   server default
    phase     the model is simply not loaded

With the guardrail off a request costs generation alone (~10 s at 0.75 MP on a
4090) and returns `"guardrail": null` rather than a fabricated pass - a result
that was never checked must not look like one that passed.

## Auth

The API is the only thing that holds secrets. Static pages cannot: anything
shipped to a browser is readable by whoever loads it, so the S3 test page
carries no credentials and no bucket keys. It collects a username and password,
exchanges them at /v1/auth/token for a short-lived bearer token, and keeps that
token in memory. Every call to the GPU box carries the token.

If the page itself must be private, that is a CloudFront function doing basic
auth in front of the bucket - not a password in the HTML.

## Deployment

    deploy/setup.sh      bare container -> full stack (~10 min cold)
    deploy/sync.sh       push code, restart the API
    deploy/connect.sh    what survived the last restart

The rented instances return on a new port with a fresh container; /workspace
persists, the venv does not.
