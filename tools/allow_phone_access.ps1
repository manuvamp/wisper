# Lets the phone keyboards reach wisper: inbound TCP 47822 (streaming), 47823 (OpenAI-compatible API), 47824 (pair.py --apk download).
# Private networks (your home Wi-Fi) and Tailscale (100.64.0.0/10) only; never public Wi-Fi. Asks for admin once.
#   Remove with: Remove-NetFirewallRule -Group wisper-phone
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Wait
    exit
}
Remove-NetFirewallRule -Group wisper-phone -ErrorAction SilentlyContinue
New-NetFirewallRule -DisplayName "wisper phone keyboard (home Wi-Fi)" -Group wisper-phone -Direction Inbound `
    -Protocol TCP -LocalPort 47822-47824 -Profile Private -Action Allow | Out-Null
New-NetFirewallRule -DisplayName "wisper phone keyboard (Tailscale)" -Group wisper-phone -Direction Inbound `
    -Protocol TCP -LocalPort 47822-47824 -RemoteAddress 100.64.0.0/10 -Action Allow | Out-Null
# mDNS: phones find this PC again by name when its Wi-Fi address changes (home network only)
New-NetFirewallRule -DisplayName "wisper phone keyboard (find PC, mDNS)" -Group wisper-phone -Direction Inbound `
    -Protocol UDP -LocalPort 5353 -Profile Private -Action Allow | Out-Null
"firewall rules added (group wisper-phone)"
