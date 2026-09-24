# Deploy ronak.sh with HTTPS

GitHub stores the source and runs the build. Cloudflare Workers Static Assets is
the deployment target and supplies DNS, TLS certificates and redirects. Namecheap
remains the registrar. Deploy the site once, use `https://ronak.sh` as its primary
address, and redirect the other owned domains to it while preserving paths and
query strings.

## Repository configuration

- `wrangler.jsonc` names the Worker `ronak-site`, uploads `dist/`, enables
  `workers.dev`, and sets the explicit single-page-application fallback.
- `.github/workflows/deploy-cloudflare.yml` runs on pushes to `main` or manual runs
  on `main`. It checks credentials, installs locked dependencies, checks repository
  and article content, builds, checks output boundaries, and deploys with Wrangler 4.
- The existing `.github/workflows/deploy.yml` continues publishing to GitHub Pages
  during migration. Disable that workflow only after Cloudflare and the redirects
  have been verified.

The configuration alone does not change any domain or publish local edits. Commit
and push the reviewed changes to run the workflow. Include the branding changes,
SVG/ICO/touch icons and any required source files in that reviewed commit; avoid
staging unrelated work wholesale.

## Account setup and first deployment

1. In Cloudflare, create an **Edit Cloudflare Workers** API token scoped to the
   intended account and domain zone. Copy that account's **Account ID**, not Zone ID.
2. In [repository Actions secrets](https://github.com/Ronaknowal/Portfolio/settings/secrets/actions),
   add `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`. Keep the token out of source
   files and chat. The workflow uses repository secrets, not environment secrets.
3. Open **Workers & Pages** and complete any account or `workers.dev` subdomain
   setup requested by Cloudflare.
4. Publish the reviewed changes to `main`. In GitHub Actions, open **Deploy to
   Cloudflare** and verify that the build and deployment succeed.
5. Open the exact `workers.dev` URL reported by the deployment. Check the home
   page, icons, `/portfolio`, `/learn`, `/articles`, and a nested lesson URL.

The deploy action installs Wrangler; no backend Worker script is required.
Local checks before publishing:

```sh
npm run check:repository
npm run check:content
npm run build
npm run check:site-build
npx --yes wrangler@4 deploy --dry-run
```

The last command validates the Cloudflare build without publishing. Wrangler's
local cache is excluded by the root `/.wrangler/` ignore rule.

## Move ronak.sh DNS to Cloudflare

1. Add `ronak.sh` to Cloudflare and review the imported DNS records against
   Namecheap. Preserve records used by email or other services. Namecheap's parking
   CNAME and parking URL redirect are not needed for this website.
2. If Namecheap email forwarding is in use, migrate that service first: copying
   its SPF TXT record alone does not keep forwarding working with custom nameservers.
3. Before switching nameservers, disable the old DNSSEC configuration and verify
   that its DS record has cleared. If DNSSEC is already off, still resolve any
   existing DNSSEC validation failure before proceeding.
4. Copy the two nameservers assigned to this zone by Cloudflare. Enter them in
   **Namecheap → Domain → Nameservers → Custom DNS**. The **Personal DNS Server**
   section is not used for this change.
5. Wait until the Cloudflare zone is **Active** and public DNS resolves correctly.

On 24 September 2026, public DNS checks found a DNSSEC validation failure for
`ronak.sh`: validated queries returned SERVFAIL, while validation-bypassed queries
returned Namecheap nameservers. This is a historical observation, not current
status or proof of a specific cause. Recheck DNS before the cutover.

## Attach ronak.sh and enable HTTPS

Once the Worker is verified and the zone is active, add this top-level property to
`wrangler.jsonc` and publish the change through the same workflow:

```json
"routes": [
  { "pattern": "ronak.sh", "custom_domain": true }
]
```

Retain the existing `name`, `compatibility_date`, `workers_dev` and `assets`
properties. Cloudflare creates the custom domain's DNS record and certificate.
Resolve any conflicting record for this exact hostname before attaching it.
Alternatively, add the Custom Domain in the Worker's **Settings → Domains & Routes**,
then record the same route in Wrangler so later deployments preserve it.

Once the certificate is active, verify `https://ronak.sh` and enable **SSL/TLS →
Edge Certificates → Always Use HTTPS**. No purchased SSL certificate is needed.
Keep `workers.dev` enabled until the domain cutover has passed verification.

`public/CNAME`, `.nojekyll` and `404.html` support the older Pages deployment.
They do not configure Workers domains. Keep `404.html`: Workers uses the explicit
SPA setting for unmatched page URLs, so no catch-all `_redirects` rule is needed.

## Redirect additional domains

Add each owned domain as its own Cloudflare zone and migrate its DNS after
preserving any existing services. Configure these aliases:

| Cloudflare zone | Redirect hostnames |
| --- | --- |
| `ronak.sh` | `www.ronak.sh` |
| `ronaksharma.me` | `ronaksharma.me`, `www.ronaksharma.me`, `portfolio.ronaksharma.me` |
| `ronaksharma.ai` | Apex and `www`, only after purchase and zone activation |

For a hostname used only for redirects, create a **proxied** A record pointing
to `192.0.2.0`. This placeholder allows Cloudflare to receive requests and apply
the rule; it is not the website's origin. Do not apply it to the primary `ronak.sh`
hostname managed by the Worker or to other services.

In **Rules → Redirect Rules**, configure each alias with a wildcard rule. Example:

- Request URL: `http*://ronaksharma.me/*`
- Target URL: `https://ronak.sh/${2}`
- Status: **301**
- **Preserve query string**: enabled

Repeat with each exact alias hostname. Ensure HTTPS certificates cover the
aliases before moving live traffic: TLS must work before an HTTPS redirect can
be returned. Test both HTTP and HTTPS, nested paths and query strings.

After DNS is stable, enable Cloudflare DNSSEC and enter the new DS details in
Namecheap. Verify DNSSEC again. After the primary domain and every existing alias
work, disable the old **Deploy to GitHub Pages** workflow in GitHub Actions.

## References

- [Cloudflare: GitHub Actions deployment](https://developers.cloudflare.com/workers/ci-cd/external-cicd/github-actions/)
- [Cloudflare: SPA routing](https://developers.cloudflare.com/workers/static-assets/routing/single-page-application/)
- [Cloudflare: Worker custom domains](https://developers.cloudflare.com/workers/configuration/routing/custom-domains/)
- [Cloudflare: full DNS setup](https://developers.cloudflare.com/dns/zone-setups/full-setup/setup/)
- [Cloudflare: redirects to another domain](https://developers.cloudflare.com/rules/url-forwarding/examples/redirect-all-another-domain/)
- [Namecheap: custom nameservers and forwarding limitations](https://www.namecheap.com/support/api/methods/domains-dns/set-custom/)
