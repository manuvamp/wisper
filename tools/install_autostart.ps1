# Registers wisper to start hidden at logon and restart if it crashes, plus the nightly learner.  Remove with:
#   Unregister-ScheduledTask -TaskName wisper -Confirm:$false; Unregister-ScheduledTask -TaskName wisper-learn -Confirm:$false
$root = Split-Path -Parent $PSScriptRoot
$py = (Get-Command pythonw.exe -ErrorAction Stop).Source
# Priority 4 matters: the Task Scheduler default (7) makes the model load take minutes.
$action = New-ScheduledTaskAction -Execute $py -Argument "-I `"$root\service\wisper.py`"" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -Priority 4
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName "wisper" -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
    -Description "wisper: local push-to-talk dictation (hold Ctrl+Shift) ($root)" -Force | Out-Null
Start-ScheduledTask -TaskName "wisper"
"registered + started"

# Nightly learner: 03:30, or as soon as the PC is on if it slept through it; low priority so it never competes.
$la = New-ScheduledTaskAction -Execute $py -Argument "-I `"$root\learn\learn.py`"" -WorkingDirectory $root
$lt = New-ScheduledTaskTrigger -Daily -At 3:30am
$ls = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2) -Priority 7
Register-ScheduledTask -TaskName "wisper-learn" -Action $la -Trigger $lt -Settings $ls -Principal $principal `
    -Description "Nightly: rebuild the vocabulary and learn recurring mishearings from the day's dictations" -Force | Out-Null
"nightly learner registered"
