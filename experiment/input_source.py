import json,time,hashlib
from pathlib import Path
import numpy as np
import torch

T=[-.05,0.,.45,.5,-.5,.5,-.5]
class LiveDepth:
    def __init__(self,root):
        import pyrealsense2 as rs
        self.root=root;self.rs=rs;self.pipeline=rs.pipeline();cfg=rs.config()
        devices=rs.context().query_devices();dev=next((d for d in devices if "D435" in d.get_info(rs.camera_info.name)),None)
        if dev is None:raise RuntimeError("Connected D435i not found; do not classify absent device as unsupported")
        serial=dev.get_info(rs.camera_info.serial_number);cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth,640,480,rs.format.z16,30);cfg.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
        profile=self.pipeline.start(cfg);self.align=rs.align(rs.stream.color)
        intr=profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K=np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]],np.float32)
        self.scale=profile.get_device().first_depth_sensor().get_depth_scale();self.realsense={"depth_scale":self.scale}
        self.archive=root/"inputs"/time.strftime("%Y%m%d-%H%M%S");self.archive.mkdir();(self.archive/"frames").mkdir()
        self.writer=(self.archive/"frames.jsonl").open("w");self.records=[]
        self.metadata={"device":dev.get_info(rs.camera_info.name),"device_serial":serial,"depth_unit":"raw_z16:uint16 SDK units; depth_m:float32 meters, post clip and edge mask","depth_scale_m_per_unit":self.scale,"intrinsics":self.K.tolist(),"alignment":"depth to color640x480; no RGB stored","camera_to_robot_pose_wxyz":T,"physics_dt":1/60,"solver_dt":.02,"voxel_size_m":.03,"integrator":"occupancy","map_initial_state":"fresh empty Mapper","map_operations":["decay_occupancy","add_depth_frame","update_esdf","cuda_synchronize","update_hashmaps"],"clipping_distance_m":.7,"edge_mask_ratio":.05,"warmup_frames":90,"initial_robot":"franka.yml retract_config","target_pose_wxyz":[.5,0,.4,0,1,0,0],"seed":0}
        for _ in range(90):self.preview=self.get_data(0)
    def has_update(self,step,global_step):return global_step%2==0
    def get_data(self,step):
        frames=self.pipeline.wait_for_frames();aligned=self.align.process(frames);f=aligned.get_depth_frame()
        if not f:raise RuntimeError("No aligned depth frame")
        color=aligned.get_color_frame();self.rgb=np.asarray(color.get_data()).copy() if color else np.zeros((480,640,3),np.uint8)
        self.raw=np.asarray(f.get_data()).copy();self.capture={"sdk_frame_number":f.get_frame_number(),"sdk_timestamp_ms":f.get_timestamp(),"sdk_timestamp_domain":str(f.get_frame_timestamp_domain()),"wall_utc_ns":time.time_ns(),"wall_monotonic_ns":time.monotonic_ns()}
        depth=self.raw.astype(np.float32);depth[depth>.7/self.scale]=0;depth*=self.scale
        self.latest={"depth":torch.from_numpy(depth),"raw_depth":depth,"raw_rgb":self.rgb,"intrinsics":torch.from_numpy(self.K.copy())}
        return self.latest
    def publish(self,step,global_step,sim_time,phase):
        idx=len(self.records);file="frames/%06d.npz"%idx;d=self.latest["depth"].numpy().copy()
        np.savez_compressed(self.archive/file,raw_z16=self.raw,depth_m=d)
        stage_file=self.root/"capture-stage.txt"
        stage=stage_file.read_text().strip() if stage_file.exists() else "EMPTY"
        record={"capture_stage":stage,"ordinal":idx,"map_step":step,"global_physics_step":global_step,"sim_time":sim_time,"phase":phase,"file":file,"depth_sha256":hashlib.sha256(d.tobytes()).hexdigest(),**self.capture}
        self.records.append(record);self.writer.write(json.dumps(record)+"\n");self.writer.flush()
    def finish(self,steps,event_file):
        self.writer.close();self.metadata.update({"total_steps":steps,"frame_count":len(self.records),"phase_events":[json.loads(x) for x in event_file.read_text().splitlines()] if event_file.exists() else []})
        self.metadata["frame_manifest_sha256"]=hashlib.sha256((self.archive/"frames.jsonl").read_bytes()).hexdigest()
        (self.archive/"metadata.json").write_text(json.dumps(self.metadata,indent=2))
        print("REPLAY_INPUT_SAVED",self.archive,flush=True)
    def stop_device(self):self.pipeline.stop()

class TapeDepth:
    def __init__(self,path):
        self.archive=Path(path);self.metadata=json.loads((self.archive/"metadata.json").read_text());self.records=[json.loads(x) for x in (self.archive/"frames.jsonl").read_text().splitlines()]
        assert hashlib.sha256((self.archive/"frames.jsonl").read_bytes()).hexdigest()==self.metadata["frame_manifest_sha256"]
        self.by_step={r["map_step"]:r for r in self.records};assert len(self.by_step)==len(self.records)
        self.K=np.array(self.metadata["intrinsics"],np.float32);self.realsense={"depth_scale":self.metadata["depth_scale_m_per_unit"]};self.preview=self._read(self.records[0]);self.used=[]
    def _read(self,r):
        with np.load(self.archive/r["file"]) as f:d=f["depth_m"].copy()
        assert hashlib.sha256(d.tobytes()).hexdigest()==r["depth_sha256"]
        return {"depth":torch.from_numpy(d),"raw_depth":d,"intrinsics":torch.from_numpy(self.K.copy())}
    def has_update(self,step,global_step):return step in self.by_step
    def get_data(self,step):
        r=self.by_step[step];self.used.append(r["ordinal"]);return self._read(r)
    def phase_at(self,step):
        phases=[e for e in self.metadata["phase_events"] if e["step"]<=step];return phases[-1]["phase"] if phases else "A"
    def publish(self,*args):pass
    def finish(self,steps,event_file):
        expected=[r["ordinal"] for r in self.records if r["map_step"]<=steps]
        assert self.used==expected,(len(self.used),len(expected))
    def stop_device(self):pass
