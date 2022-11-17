# CUDA Heat Conduction Demo

Demo im Rahmen der Vorlesung _Parallele Systeme_ an der
[Hochschule Karlsruhe](https://www.h-ka.de/)

![Screenshot](screenshot.png)

## Benutzung

- `[F1]` Zeige/Verstecke Settings Fenster
- `[Rechtsklick]` Heißer Punkt hinzufügen

## Kompilieren

```Shell
mkdir build && cd build # Build-Verzeichnis erstellen
cmake ..                # Build Dateien erzeugen
cmake --build .         # Programm bauen
```

## Setup

Bestandteile:

- SSH mit X-Forwarding
- [VcXsrv](https://sourceforge.net/projects/vcxsrv/) als X-Server für Windows
    - Die Einstellungen die ich verwende: [VcXsrv.xlaunch](VcXsrv.xlaunch)

### SSH Konfiguration

`$HOME/.ssh/config` bzw. `%USERPROFILE%\.ssh\config` für Windows:

```
Host hska
    HostName login.hs-karlsruhe.de
    HostkeyAlgorithms ssh-rsa,ssh-dss
    StrictHostKeyChecking no

Host parallele-systeme
    HostName 10.154.35.141
    ProxyJump hska
    ForwardX11 yes
    ForwardX11Trusted yes
```

Verbindung zum Server: `ssh BENUTZERNAME@parallele-systeme`
