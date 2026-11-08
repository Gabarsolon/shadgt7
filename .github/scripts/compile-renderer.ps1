$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

$source = (Get-Location).Path
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ($LASTEXITCODE -or !$vs) { throw 'Visual Studio C++ x64 tools are required' }
& (Join-Path $vs 'Common7/Tools/Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
if (!$env:VCINSTALLDIR) { throw 'Visual Studio developer environment was not initialized' }
Set-Location $source

$tools = Join-Path $env:RUNNER_TEMP 'renderer-compile-tools'
New-Item -ItemType Directory -Path $tools -Force | Out-Null
$archive = Join-Path $tools 'llvm-23.1.3.tar.xz'
Invoke-WebRequest -Uri 'https://github.com/llvm/llvm-project/releases/download/llvmorg-23.1.3/clang%2Bllvm-23.1.3-x86_64-pc-windows-msvc.tar.xz' -OutFile $archive
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne '9ee6a2192fbb30c776415af529927f76ca6b5d1d55699bb1315f9878e610b5b8') {
    throw 'LLVM archive SHA-256 mismatch'
}
$sevenZip = Join-Path $env:ProgramFiles '7-Zip/7z.exe'
Invoke-Checked $sevenZip @('x', $archive, "-o$tools", '-y', '-bso0', '-bsp0')
Invoke-Checked $sevenZip @('x', (Join-Path $tools 'llvm-23.1.3.tar'), "-o$tools", '-y', '-bso0', '-bsp0')
$llvm = Join-Path $tools 'clang+llvm-23.1.3-x86_64-pc-windows-msvc'
if (!(Test-Path -LiteralPath (Join-Path $llvm 'bin/clang-cl.exe'))) { throw 'LLVM compiler missing' }
$builtins = Get-ChildItem "$llvm/lib" -Recurse -Filter clang_rt.builtins-x86_64.lib | Select-Object -First 1
if (!$builtins) { throw 'LLVM compiler-rt builtins are required' }
Invoke-Checked 'python' @('-m', 'pip', 'install', '--disable-pip-version-check', 'cmake==3.31.6', 'ninja==1.11.1.4')
$scripts = & python -c "import sysconfig; print(sysconfig.get_path('scripts'))"
if ($LASTEXITCODE) { throw 'Python scripts path lookup failed' }
$env:PATH = "$llvm/bin;$scripts;$env:PATH"
$env:LIB = "$($builtins.DirectoryName);$env:LIB"

Invoke-Checked 'cmake' @('-S', '.', '-B', 'Build/pr-compile', '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
    '-DCMAKE_C_COMPILER=clang-cl', '-DCMAKE_CXX_COMPILER=clang-cl', '-DCMAKE_LINKER=lld-link',
    '-DHAVE_ENDIAN_H=OFF', '-DSHADGT_RYTY_GPU=ON', '-DENABLE_DISCORD_RPC=ON')
Invoke-Checked 'cmake' @('--build', 'Build/pr-compile', '--target', 'ryty_gpu', '--parallel', '2')
if (@(Get-ChildItem 'Build/pr-compile' -Recurse -Filter libSceVideoOut.prx).Count -ne 1) {
    throw 'Expected exactly one renderer output'
}
