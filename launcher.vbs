Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")

strPath = WScript.ScriptFullName
strFolder = FSO.GetParentFolderName(strPath)

pythonExe = strFolder & "\.venv\Scripts\pythonw.exe"
If Not FSO.FileExists(pythonExe) Then
    pythonExe = strFolder & "\.venv\Scripts\python.exe"
End If

launcherPy = strFolder & "\launcher.py"
WshShell.CurrentDirectory = strFolder
WshShell.Run """" & pythonExe & """ """ & launcherPy & """", 0, False
