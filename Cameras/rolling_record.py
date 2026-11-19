"""Record closed SVO parts while keeping the live camera open between swings."""
import csv
import json
import os
from pathlib import Path
import shutil
import time


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str))
    os.replace(str(tmp), str(path))


def record_segments(cam, cfg, output, codec='h265', segment_s=8.):
    """No marker detection here. A seal request closes the completed swing's part.

    Parts also rotate every eight seconds. Their boundaries may lose frames during
    encoder finalisation; the decoded timestamp audit reports those gaps honestly.
    """
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {'camera': cam.info(), 'calibration': cam.calibration().to_dict(),
                'gravity': cam.estimate_gravity(.5), 'config': cfg.to_dict(), 'codec': codec}
    manifest = (folder/'segments.jsonl').open('a')
    total = index = closed_count = 0
    first = latest = None
    part_first = part_last = None
    part_frames = 0
    frame_file = writer = path = None
    sequence = 0
    last_check = last_status = 0.

    def status(running=True):
        free = shutil.disk_usage(str(folder)).free
        atomic_json(folder/'recorder_status.json', {
            'running': running, 'first_ns': first, 'latest_ns': latest,
            'frames': total, 'closed_parts': closed_count,
            'dropped_reported': cam.dropped_frames(), 'free_disk_bytes': free})
        if running and free < 256*1024*1024:
            raise RuntimeError('Disk nearly full; continuous recording stopped and raw files kept')

    def start_part():
        nonlocal index, path, frame_file, writer, part_first, part_last, part_frames
        index += 1
        path = folder/('camera_%06d.svo2' % index)
        cam.start_recording(str(path), codec=codec)
        atomic_json(str(path)+'.meta.json', metadata)
        frame_file = path.with_suffix('.frames.csv').open('w', newline='')
        writer = csv.writer(frame_file)
        writer.writerow(['frame_id', 'ts_ns_image'])
        part_first = part_last = None
        part_frames = 0

    def seal():
        nonlocal frame_file, closed_count
        if frame_file is None:
            return
        cam.stop_recording()  # manifest readers only see fully closed files
        frame_file.close()
        frame_file = None
        if part_frames:
            manifest.write(json.dumps({'path': path.name, 'first_ns': part_first,
                                       'last_ns': part_last, 'frames': part_frames})+'\n')
            manifest.flush()
            closed_count += 1

    try:
        start_part()
        print('Recording continuously: %s @ %.0f FPS. Waiting for IMU triggers.' % (
            metadata['camera']['resolution'], metadata['camera']['fps']), flush=True)
        while True:
            ts = cam.grab_timestamp_only()
            if ts is None:
                break
            total += 1
            part_frames += 1
            latest = part_last = ts
            if first is None:
                first = ts
            if part_first is None:
                part_first = ts
            writer.writerow([total, ts])
            now = time.monotonic()
            requested = False
            if now-last_check >= .1:
                last_check = now
                try:
                    request = json.loads((folder/'seal_request.json').read_text())
                    if request['sequence'] > sequence and ts >= request['through_ns']:
                        sequence = request['sequence']
                        requested = True
                except FileNotFoundError:
                    pass
            if requested or (ts-part_first)*1e-9 >= segment_s:
                seal()
                start_part()
            if now-last_status >= .25:
                last_status = now
                status()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            seal()
            status(False)
        finally:
            manifest.close()
            cam.close()
    print('Saved %d acquired frames in %s' % (total, folder), flush=True)
    return 0
