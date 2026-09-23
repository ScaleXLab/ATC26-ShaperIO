#!/usr/bin/env python3
"""Format and bind the configured PM9A3 experiment device."""
import argparse
import json
import re
import subprocess
from pathlib import Path

BDF = '0000:27:00.0'
SERIAL = 'S6CKNT0W914900'
MODEL = 'SAMSUNG MZQL27T6HBLA-00A07'
PCI = Path('/sys/bus/pci/devices') / BDF
MOUNT = Path('/mnt/shaperio-ae-pm9a3')
RAW = '/dev/libnvm0'


def configure(config):
    global BDF, SERIAL, MODEL, PCI, MOUNT, RAW
    BDF, SERIAL, MODEL = (config[k] for k in ('device_bdf', 'device_serial', 'device_model'))
    if not re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', BDF):
        raise ValueError('Invalid PCI address')
    if not SERIAL or not MODEL or config['namespace_id'] != 1:
        raise ValueError('Set the device identity and namespace 1')
    file = Path(config['gds_file'])
    if not file.is_absolute() or file.name != 'core.bin' or file.parent == Path('/'):
        raise ValueError('gds_file must be an absolute mount-directory/core.bin path')
    if config['gds_offset_bytes'] != 0:
        raise ValueError('GDS file offset must be zero')
    PCI, MOUNT, RAW = Path('/sys/bus/pci/devices') / BDF, file.parent, config['controller']


def run(*args):
    print('+', *map(str, args), flush=True)
    result = subprocess.run(list(map(str, args)), text=True, capture_output=True)
    if result.stderr:
        print(result.stderr, end='', flush=True)
    result.check_returncode()
    return result.stdout


def identify():
    nodes = list((PCI / 'nvme').glob('nvme*'))
    if len(nodes) != 1:
        raise RuntimeError('Expected one kernel NVMe controller at configured PCI address')
    controller = Path('/dev') / nodes[0].name
    info = json.loads(run('nvme', 'id-ctrl', controller, '-o', 'json'))
    if info['sn'].strip() != SERIAL or info['mn'].strip() != MODEL:
        raise RuntimeError('PM9A3 identity mismatch')
    print(json.dumps({'bdf': BDF, 'model': info['mn'].strip(), 'serial': info['sn'].strip()}), flush=True)
    namespaces = json.loads(run('nvme', 'list-ns', controller, '-a', '-o', 'json'))
    active = [row['nsid'] for row in namespaces['nsid_list']]
    if active != [1]:
        raise RuntimeError(f'Unexpected namespaces: {namespaces}')
    device = Path(str(controller) + 'n1')
    if not device.exists():
        raise RuntimeError('Namespace device absent')
    return device


def unmount(device):
    tree = json.loads(run('lsblk', '--json', '--output', 'PATH,MOUNTPOINTS', device))
    def children(node):
        for child in node.get('children', []):
            yield from children(child)
        yield node
    for node in children(tree['blockdevices'][0]):
        for target in node.get('mountpoints', []):
            if target:
                print(run('umount', target), end='')
    result = subprocess.run(['findmnt', '-rn', '-S', str(device), '-o', 'TARGET'],
                            text=True, capture_output=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    for target in result.stdout.splitlines():
        print(run('umount', target), end='')


def bind(driver):
    old = (PCI / 'driver').resolve().name if (PCI / 'driver').exists() else None
    if old == driver:
        return
    if old:
        (PCI / 'driver' / 'unbind').write_text(BDF)
    try:
        (Path('/sys/bus/pci/drivers') / driver / 'bind').write_text(BDF)
    except BaseException:
        if old:
            (Path('/sys/bus/pci/drivers') / old / 'bind').write_text(BDF)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'kernel', 'raw', 'format-raw', 'format-gds', 'mount-existing'])
    parser.add_argument('--allow-destroy', action='store_true')
    parser.add_argument('--file-bytes', type=int, default=1073741824)
    parser.add_argument('--secure-erase', type=int, choices=[0, 1], default=0)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/device.json')
    args = parser.parse_args()
    configure(json.loads(args.config.read_text()))
    if args.action == 'kernel':
        bind('nvme')
        print(identify())
        return
    if (PCI / 'driver').resolve().name != 'nvme':
        if args.action not in ('format-raw', 'format-gds') or not args.allow_destroy:
            raise RuntimeError('Restore kernel driver before this operation')
        bind('nvme')
    device = identify()
    if args.action == 'inspect':
        print(device)
        return
    if args.action in ('raw', 'format-raw', 'format-gds') and not args.allow_destroy:
        raise RuntimeError('--allow-destroy is required for exclusive raw storage use')
    unmount(device)
    if args.action.startswith('format-'):
        # Format is required between write conditions; secure erase is optional.
        print(run('nvme', 'format', device, '--namespace-id=1', '--lbaf=0',
                  f'--ses={args.secure_erase}', '--force'), end='')
        print(run('udevadm', 'settle'), end='')
    if args.action in ('raw', 'format-raw'):
        driver = Path('/sys/bus/pci/drivers/libnvm helper')
        others = [p.name for p in driver.iterdir()
                  if re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', p.name)
                  and p.name != BDF]
        if others:
            raise RuntimeError(f'libnvm already owns other controllers: {others}')
        bind('libnvm helper')
        print(run('udevadm', 'settle'), end='')
        print(run('ls', '-l', '/sys/class/libnvm helper'), end='')
        nodes = list(Path('/sys/class/libnvm helper').glob('libnvm*'))
        if len(nodes) != 1 or nodes[0].name != Path(RAW).name or not Path(RAW).exists():
            raise RuntimeError(f'Expected the single libnvm controller at {RAW}')
    else:
        if args.action == 'format-gds':
            print(run('mkfs.ext4', '-F', '-m', '0', '-T', 'largefile4', '-E',
                      'lazy_itable_init=0,lazy_journal_init=0,nodiscard', '-L', 'shaperio-ae', device), end='')
        MOUNT.mkdir(parents=True, exist_ok=True)
        print(run('mount', '-t', 'ext4', '-o', 'noatime,data=ordered', device, MOUNT), end='')
        print(run('findmnt', MOUNT), end='')
        if args.action == 'format-gds':
            print(run('fallocate', '-l', args.file_bytes, MOUNT / 'core.bin'), end='')


if __name__ == '__main__':
    main()
