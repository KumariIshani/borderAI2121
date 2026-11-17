import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  FileVideo,
  FileImage,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Download,
  Layers,
  Users,
  ShieldAlert,
  Crosshair,
  Truck
} from 'lucide-react';
import { AnalysisJob } from '@/types';

export const UploadAnalyzePage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [job, setJob] = useState<AnalysisJob | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [isLoadingSample, setIsLoadingSample] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelected = (selectedFile: File) => {
    setFile(selectedFile);
    setJob(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(URL.createObjectURL(selectedFile));
  };

  const startAnalysis = async () => {
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Analysis submission failed');
      const data = await res.json();
      const jobId = data.job_id;

      // Poll job status
      const pollInterval = setInterval(async () => {
        try {
          const pollRes = await fetch(`/api/jobs/${jobId}`);
          if (pollRes.ok) {
            const jobData: AnalysisJob = await pollRes.json();
            setJob(jobData);

            if (jobData.status === 'completed' || jobData.status === 'error') {
              clearInterval(pollInterval);
              setIsUploading(false);
            }
          }
        } catch (err) {
          console.error('Error polling job:', err);
        }
      }, 800);
    } catch (err: any) {
      alert(err.message || 'Error uploading file');
      setIsUploading(false);
    }
  };

  // Quick sample loader
  const loadSample = async (samplePath: string, fileName: string, type: string) => {
    try {
      setIsLoadingSample(type);
      const res = await fetch(samplePath);
      if (!res.ok) throw new Error('Sample could not be fetched');
      const blob = await res.blob();
      const sampleFile = new File([blob], fileName, { type: blob.type || (type === 'video' ? 'video/mp4' : 'image/jpeg') });
      handleFileSelected(sampleFile);
    } catch (e: any) {
      alert(`Could not load sample file: ${e.message || e}`);
    } finally {
      setIsLoadingSample(null);
    }
  };

  const isVideo = file?.type.startsWith('video') || file?.name.match(/\.(mp4|avi|mov|mkv|webm)$/i);

  return (
    <div className="space-y-6">
      {/* Detection Color Legend & Classification Guide */}
      <div className="glass-card p-4 border border-white/10 bg-[#0E1318]">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 mb-3">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-orange-400 shrink-0" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-white">Detection Color Legend & Threat Key</h3>
          </div>
          <span className="text-[11px] font-mono text-[#9CA3AF]">Clean Tactical Surveillance (Zones 1-3 Disabled)</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          <div className="flex items-center gap-2.5 p-2 rounded-lg bg-red-500/10 border border-red-500/30">
            <span className="w-3.5 h-3.5 rounded-full bg-red-500 shrink-0 shadow-[0_0_8px_rgba(239,68,68,0.7)]" />
            <div>
              <div className="text-xs font-bold text-red-400">ARMED PERSON</div>
              <div className="text-[10px] text-[#9CA3AF] font-mono leading-tight">Hostile Carrying Firearm</div>
            </div>
          </div>
          <div className="flex items-center gap-2.5 p-2 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
            <span className="w-3.5 h-3.5 rounded-full bg-yellow-400 shrink-0 shadow-[0_0_8px_rgba(250,204,21,0.7)]" />
            <div>
              <div className="text-xs font-bold text-yellow-400">WEAPON</div>
              <div className="text-[10px] text-[#9CA3AF] font-mono leading-tight">Rifle, Handgun, Gun</div>
            </div>
          </div>
          <div className="flex items-center gap-2.5 p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30">
            <span className="w-3.5 h-3.5 rounded-full bg-cyan-400 shrink-0 shadow-[0_0_8px_rgba(34,211,238,0.7)]" />
            <div>
              <div className="text-xs font-bold text-cyan-400">PERSON</div>
              <div className="text-[10px] text-[#9CA3AF] font-mono leading-tight">Human / Personnel</div>
            </div>
          </div>
          <div className="flex items-center gap-2.5 p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30">
            <span className="w-3.5 h-3.5 rounded-full bg-emerald-400 shrink-0 shadow-[0_0_8px_rgba(52,211,153,0.7)]" />
            <div>
              <div className="text-xs font-bold text-emerald-400">VEHICLE</div>
              <div className="text-[10px] text-[#9CA3AF] font-mono leading-tight">Car, Truck, Transport</div>
            </div>
          </div>
          <div className="flex items-center gap-2.5 p-2 rounded-lg bg-orange-500/10 border border-orange-500/30">
            <span className="w-3.5 h-3.5 rounded-full bg-orange-500 shrink-0 shadow-[0_0_8px_rgba(249,115,22,0.7)]" />
            <div>
              <div className="text-xs font-bold text-orange-400">MILITARY VEHICLE</div>
              <div className="text-[10px] text-[#9CA3AF] font-mono leading-tight">Combat Armor / APC</div>
            </div>
          </div>
        </div>
      </div>

      {/* Upload Box */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-lg font-bold text-white tracking-wide">Upload Surveillance Media for Neural Analysis</h2>
            <p className="text-xs text-[#9CA3AF] font-mono mt-0.5">
              Supports JPEG, PNG, MP4, AVI, MOV &middot; Direct YOLOv8 detection + armed proximity linking
            </p>
          </div>
          {/* Sample quick buttons */}
          <div className="flex items-center gap-2">
            <button
              disabled={isLoadingSample !== null || isUploading}
              onClick={() => loadSample('/media/screenshots/screenshot_1.jpg', 'sample_border.jpg', 'image')}
              className="px-2.5 py-1 text-xs font-mono text-[#E5E7EB] bg-white/5 hover:bg-white/10 disabled:opacity-50 border border-white/10 rounded-lg transition-colors flex items-center gap-1.5"
            >
              {isLoadingSample === 'image' && <Loader2 className="w-3 h-3 animate-spin" />}
              <span>Load Sample Image</span>
            </button>
            <button
              disabled={isLoadingSample !== null || isUploading}
              onClick={() => loadSample('/media/cam01.mp4', 'sample_patrol.mp4', 'video')}
              className="px-2.5 py-1 text-xs font-mono text-orange-400 bg-orange-500/10 hover:bg-orange-500/20 disabled:opacity-50 border border-orange-500/30 rounded-lg transition-colors flex items-center gap-1.5"
            >
              {isLoadingSample === 'video' && <Loader2 className="w-3 h-3 animate-spin" />}
              <span>Load Sample Video</span>
            </button>
          </div>
        </div>

        {/* Dropzone */}
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
            dragActive
              ? 'border-orange-500 bg-orange-500/10'
              : 'border-[rgba(255,255,255,0.12)] hover:border-orange-500/50 bg-[#0C1014]'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,video/*"
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                handleFileSelected(e.target.files[0]);
              }
            }}
          />

          <div className="flex flex-col items-center justify-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-orange-500/10 border border-orange-500/25 flex items-center justify-center text-orange-400">
              <UploadCloud className="w-6 h-6" />
            </div>
            <div>
              <p className="text-sm font-medium text-white">
                {file ? file.name : 'Click to select or drag and drop surveillance media'}
              </p>
              <p className="text-xs text-[#9CA3AF] font-mono mt-1">
                {file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : 'Images or videos up to 100 MB'}
              </p>
            </div>
          </div>
        </div>

        {/* Action Button & Progress */}
        {file && (
          <div className="mt-4 flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-[#9CA3AF]">
                Selected: <strong className="text-white">{file.name}</strong>
              </span>
              <button
                disabled={isUploading}
                onClick={startAnalysis}
                className="px-5 py-2 rounded-xl bg-orange-500 hover:bg-orange-600 disabled:opacity-50 text-black font-bold text-xs tracking-wider uppercase font-mono shadow-[0_0_15px_rgba(249,115,22,0.3)] transition-all flex items-center gap-2"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Processing Neural Inference...</span>
                  </>
                ) : (
                  <span>Run YOLOv8 Model Inference</span>
                )}
              </button>
            </div>

            {/* Progress Bar */}
            {isUploading && (
              <div className="w-full bg-[#151C22] rounded-full h-2 overflow-hidden border border-[rgba(255,255,255,0.08)]">
                <div
                  className="bg-orange-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${job?.progress || 15}%` }}
                />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Results View when completed */}
      {job?.status === 'completed' && job.result && (
        <div className="space-y-6">
          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="glass-card p-4">
              <div className="text-xs font-mono text-[#9CA3AF]">
                {job.result.media_type === 'video' ? 'Unique Subjects' : 'Total Entities'}
              </div>
              <div className="text-2xl font-mono font-bold text-white mt-1">
                {job.result.total_entities}
              </div>
              {((job.result.summary as any).peak_concurrent ?? 0) > 0 && (
                <div className="text-[11px] font-mono text-orange-400 mt-1">
                  Peak: {(job.result.summary as any).peak_concurrent} active in frame
                </div>
              )}
            </div>
            <div className="glass-card p-4">
              <div className="text-xs font-mono text-cyan-400">Persons</div>
              <div className="text-2xl font-mono font-bold text-white mt-1">
                {job.result.summary.persons}
              </div>
              <div className="text-[11px] font-mono text-[#9CA3AF] mt-1">
                {job.result.media_type === 'video' ? 'Unique tracked' : 'In image'}
              </div>
            </div>
            <div className="glass-card p-4">
              <div className="text-xs font-mono text-red-400">Armed Persons</div>
              <div className="text-2xl font-mono font-bold text-red-400 mt-1">
                {job.result.summary.armed_persons}
              </div>
              <div className="text-[11px] font-mono text-[#9CA3AF] mt-1">
                Critical threat
              </div>
            </div>
            <div className="glass-card p-4">
              <div className="text-xs font-mono text-emerald-400">Vehicles</div>
              <div className="text-2xl font-mono font-bold text-white mt-1">
                {job.result.summary.vehicles + ((job.result.summary as any).military_vehicles || 0)}
              </div>
              <div className="text-[11px] font-mono text-[#9CA3AF] mt-1">
                Ground transport
              </div>
            </div>
            <div className="glass-card p-4">
              <div className="text-xs font-mono text-yellow-400">Weapons</div>
              <div className="text-2xl font-mono font-bold text-white mt-1">
                {job.result.summary.weapons}
              </div>
              <div className="text-[11px] font-mono text-[#9CA3AF] mt-1">
                Firearms & rifles
              </div>
            </div>
          </div>

          {/* Media Output & Entity Table */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Output Display */}
            <div className="lg:col-span-2 glass-card p-5">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-base font-bold text-white">Annotated Detection Output</h3>
                <a
                  href={job.result.result_url}
                  download
                  className="flex items-center gap-1.5 px-3 py-1 rounded bg-white/5 hover:bg-white/10 text-xs font-mono text-[#E5E7EB] border border-white/10 transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download File</span>
                </a>
              </div>

              <div className="rounded-xl overflow-hidden bg-black border border-[rgba(255,255,255,0.08)]">
                {job.result.media_type === 'video' ? (
                  <video
                    key={job.result.result_url}
                    src={job.result.result_url}
                    controls
                    autoPlay
                    muted
                    playsInline
                    loop
                    className="w-full aspect-video object-contain bg-black"
                  />
                ) : (
                  <img
                    src={job.result.result_url}
                    alt="Annotated detection output"
                    className="w-full max-h-[500px] object-contain mx-auto"
                  />
                )}
              </div>
            </div>

            {/* Entity Detections Table */}
            <div className="glass-card p-5 flex flex-col justify-between">
              <div>
                <h3 className="text-base font-bold text-white mb-1">
                  {job.result.media_type === 'video' ? 'Tracked Subjects' : 'Detected Objects'}
                </h3>
                <p className="text-xs text-[#9CA3AF] font-mono mb-4">
                  {job.result.media_type === 'video' ? 'Persistent entity IDs & threat classification' : 'Class and confidence breakdown'}
                </p>

                <div className="overflow-y-auto max-h-[420px] divide-y divide-[rgba(255,255,255,0.06)]">
                  {job.result.boxes && job.result.boxes.length > 0 ? (
                    job.result.boxes.map((b, idx) => {
                      const cName = (b.class || (b as any).class_name || '').toLowerCase();
                      const isArmed = cName.includes('armed') || b.armed;
                      const textColor = isArmed
                        ? 'text-red-400'
                        : cName.includes('weapon')
                        ? 'text-yellow-400'
                        : cName.includes('vehicle')
                        ? 'text-emerald-400'
                        : 'text-cyan-400';

                      const trackId = (b as any).track_id;
                      const titleLabel = trackId ? `#${trackId} ${b.class.replace('_', ' ').toUpperCase()}` : b.class.replace('_', ' ').toUpperCase();
                      const subLabel = isArmed ? 'Armed Hostile' : cName.includes('weapon') ? 'Firearm' : cName.includes('vehicle') ? 'Vehicle' : 'Personnel';

                      return (
                        <div key={idx} className="py-2.5 flex items-center justify-between text-xs font-mono">
                          <div>
                            <span className={`font-bold ${textColor}`}>
                              {titleLabel}
                            </span>
                            <div className="text-[11px] text-[#9CA3AF] mt-0.5">{subLabel}</div>
                          </div>
                          <span className="px-2 py-0.5 rounded bg-white/5 border border-white/10 text-orange-400 font-bold">
                            {Math.round(((b as any).confidence ?? (b as any).conf ?? 0) * 100)}%
                          </span>
                        </div>
                      );
                    })
                  ) : (
                    <div className="text-xs text-[#9CA3AF] font-mono py-8 text-center">
                      Processed {job.result.frames_processed || 1} frames. Overall counts reflected in summary.
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error View */}
      {job?.status === 'error' && (
        <div className="glass-card p-5 border border-red-500/40 bg-red-500/10 rounded-xl flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <h4 className="text-sm font-bold text-red-400">Analysis Failed</h4>
            <p className="text-xs text-[#E5E7EB] font-mono mt-1">
              {job.error || 'The model encountered an error while processing the uploaded file. Please check file format.'}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

