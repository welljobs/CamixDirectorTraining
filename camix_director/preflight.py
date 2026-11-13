"""Exhaustive fast preflight with actionable field-level errors, no training side effects."""
from pathlib import Path
import math
from .config import ModelConfig
from .storage import metadata_only,dataset_paths


def check_training(root,payload):
    root=Path(root).resolve();errors=[];warnings=[];summary={}
    def error(field,message):errors.append(dict(field=field,message=message))
    try:
        config=ModelConfig(**payload.get('config',{}));summary['config']=config.to_dict()
    except Exception as exc:error('config-json','模型配置无效：'+str(exc));config=None
    try:
        data=(root/payload.get('data','')).resolve()
        if not data.is_relative_to(root):raise ValueError('路径不在工作区')
        paths=dataset_paths(data,verify=False)
        if not paths:raise ValueError('目录中没有数据包')
    except Exception as exc:error('train-data','请选择含数据包的目录或冻结版本：'+str(exc));paths=[]
    splits={};ids={};weak=False;synthetic=False;pairs_unknown=0
    for path in paths:
        try:
            meta=metadata_only(path);part=meta.get('split');sid=meta.get('session_id');splits[part]=splits.get(part,0)+1
            if sid in ids and ids[sid]!=part:error('train-data',f'{sid} 跨越训练/验证/测试划分')
            ids[sid]=part
            if config:
                for key in ('max_cameras','image_size','sample_rate','audio_samples','decision_hz'):
                    if meta.get('config',{}).get(key)!=getattr(config,key):error('config-json',f'{path.name} 的 {key} 与数据不一致，需重新准备数据')
            if meta.get('source')=='synthetic':synthetic=True
            if meta.get('preview_only'):error('train-data','这是未标注回放包，不能通过勾选弱监督用于训练；请先完成偏好标注和审核。')
            elif meta.get('source')=='real' and not meta.get('human_reviewed'):weak=True
            if meta.get('source')=='real' and not meta.get('alignment_verified'):error('train-data',f'{path.name} 未确认时钟对齐')
            if meta.get('human_reviewed') and not meta.get('review'):warnings.append(f'{path.name}: 旧格式人审自声明，无版本化审核证据')
        except Exception as exc:error('train-data',f'{path.name}: {exc}')
    if not(splits.get('train') and splits.get('validation')):error('train-data','训练需要不同原始场次的 train 与 validation')
    if weak and payload.get('allow_weak') is not True:error('weak','这批数据含弱标签。请勾选“我确认使用弱监督标签”，不是把人审字段改成true。')
    if synthetic and payload.get('allow_synthetic') is not True:error('synthetic','这批数据含合成素材；仅检查链路时勾选“允许合成数据”。')
    for name,field,default,lo,hi in [('epochs','epochs',30,1,10000),('batch_size','batch',2,1,64),('sequence_steps','sequence',16,2,512),('burn_in','burn',4,0,511),('accumulation_steps','accumulation',1,1,64),('patience','patience',0,0,1000)]:
        value=payload.get(name,default)
        if isinstance(value,bool) or not isinstance(value,int) or not lo<=value<=hi:error(field,f'{name} 必须为 {lo}～{hi} 整数')
    if isinstance(payload.get('burn_in',4),int) and isinstance(payload.get('sequence_steps',16),int) and payload.get('burn_in',4)>=payload.get('sequence_steps',16):error('burn','记忆预热必须短于序列长度')
    mode=payload.get('mode','new')
    if mode not in ('new','fine_tune','resume'):error('train-mode','请选择从头、微调或完整恢复')
    if mode!='new' and not payload.get('parent_checkpoint'):error('train-parent','请选择父检查点')
    warnings.append('当前按需缓存会把压缩数据展开到磁盘；内存不再全量载入，但需足够缓存空间。')
    summary.update(splits=splits,weak_labels=weak,synthetic=synthetic,mode=mode)
    # De-duplicate repeated per-pack media configuration errors.
    unique=[];seen=set()
    for item in errors:
        key=(item['field'],item['message'])
        if key not in seen:seen.add(key);unique.append(item)
    return dict(passed=not unique,errors=unique,warnings=list(dict.fromkeys(warnings)),summary=summary,training_started=False)
