; Inno Setup script for WhisperNow Windows Installer
; Creates a proper Windows installer with Start Menu shortcuts and uninstaller

#define MyAppName "WhisperNow"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "WhisperNow"
#define MyAppURL "https://github.com/whispernow/whispernow"
#define MyAppExeName "whispernow.exe"

[Setup]
; Unique application identifier
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
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
; Privilege requirements - install for current user by default
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "startupicon"; Description: "Start WhisperNow when Windows starts"; GroupDescription: "Startup:"; Flags: unchecked

[Files]
; Install all files from the PyInstaller dist folder
Source: "dist\whispernow\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userstartup}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: startupicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up any settings/cache files on uninstall (optional)
Type: filesandordirs; Name: "{localappdata}\WhisperNow"
