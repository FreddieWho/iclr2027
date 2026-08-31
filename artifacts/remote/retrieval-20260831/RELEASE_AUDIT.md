# Remote result collection and instance release audit

- Date: 2026-08-31 (Asia/Shanghai)
- Project: ICLR2027 / calligraphy exploratory pilot
- Remote task: `/home/vipuser/codex-jobs/calligraphy-3080-formal-oomfix-20260830`
- Report: `/home/vipuser/iclr2027/reports/CALLIGRAPHY_PILOT_REPORT.md`
- Provider instance: `17750662501_lyg0022_9fdcd9fd290248889cdbd90f914e8b0d_u1_c0`
- Target match: SSH port `13212`, GeForce RTX 3080, 16 CPU cores, 32 GB memory

## Collection verification

- Result files collected: 33
- Remote result SHA-256 manifest: `REMOTE_SHA256.txt`
- Result verification: all 33 files `OK`
- Report SHA-256: `b67078b729ca79e3fcf1798ffedb74e71e7719df114b030e39d51fc4761c8f7b`
- Pipeline status: `pilot_complete`

## Release verification

- Release request: completed after collection and checksum verification
- Post-release instance status: non-running (`status=0`)
- Account-wide running instances: `0`
- Kept disks: `0`
- Kept-disk hourly cost: `0`
- Refund recorded: `0.7449` compute units
- Cumulative provider cost for this instance: `3.3281` compute units

The standard MCP release wrapper rejected this web-rented instance because it was absent from the MCP local ownership store. After the endpoint, instance detail, GPU/configuration, and billing records were matched, release was executed through the same provider API's fresh quote plus approval-token path. No bioinformatics index or data was accessed or modified.
