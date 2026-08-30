@echo off
rem Windows-gnu builds need the MSYS2 MinGW binutils (gcc, collect2, ld, as,
rem dlltool) reachable from this linker's own process. The calling shell's PATH
rem is not under our control, so establish it here before invoking gcc.
set "PATH=C:\msys64\mingw64\bin;C:\Windows\system32;C:\Windows"
"C:\msys64\mingw64\bin\gcc.exe" %*