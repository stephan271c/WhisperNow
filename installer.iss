; Inno Setup script for WhisperNow Windows Installer
; Creates a proper Windows installer with Start Menu shortcuts and uninstaller

#define MyAppName "WhisperNow"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "WhisperNow"
#define MyAppURL "https://github.com/whispernow/whispernow"
#define MyAppExeName "WhisperNow.exe"

[Setup]
; Unique application identifier - NEVER change this after first release
AppId={{13AD1DC3-955C-4618-86E0-36B01ABF13EF}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
; Output settings - the installer will be created in the Output directory
OutputDir=installer-output
OutputBaseFilename=WhisperNow-Setup-{#MyAppVersion}
; Compression settings
Compression=lzma2
SolidCompression=yes
; Windows version requirements
MinVersion=10.0
; Installer appearance
WizardStyle=modern
; Privilege requirements - install for current user by default (no UAC prompt)
; App installs to: C:\Users\<User>\AppData\Local\Programs\WhisperNow
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; 64-bit architecture settings (required for PyTorch/CUDA apps)
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "startupicon"; Description: "Start WhisperNow when Windows starts"; GroupDescription: "Startup:"; Flags: unchecked

[Files]
; Install all files from the Briefcase build folder
Source: "build\whispernow\windows\app\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userstartup}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: startupicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

; Note: Previous [UninstallDelete] section removed.
; User data in {localappdata}\WhisperNow (settings, models) is preserved on uninstall.
; This allows users to reinstall without losing their configuration or re-downloading models.
