export interface Camera {
  id: string;
  name: string;
  location: string;
  status: 'ONLINE' | 'OFFLINE' | 'DEGRADED';
  mode: 'REPLAY' | 'LIVE';
  fps: number;
  resolution: string;
  stream_url: string;
  annotated_url: string;
  poster_url: string;
  threat_level: 'NORMAL' | 'ELEVATED' | 'CRITICAL';
  active_alerts_count: number;
}

export interface BoxDetection {
  class: 'person' | 'armed_person' | 'weapon' | 'vehicle' | 'military_vehicle';
  class_id?: number;
  confidence: number;
  bbox: [number, number, number, number];
  armed?: boolean;
  zone: string;
  zone_level: 'RESTRICTED' | 'HIGH' | 'MEDIUM' | 'SAFE';
}

export interface FrameDetection {
  frame: number;
  time: number;
  boxes: BoxDetection[];
}

export interface Alert {
  id: string;
  time: string;
  camera: string;
  camera_id?: string;
  class: string;
  confidence: number;
  zone: string;
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  message: string;
  bbox?: number[];
}

export interface DashboardStats {
  kpis: {
    total_detections: number;
    persons: number;
    weapons: number;
    vehicles: number;
    military_vehicles: number;
    armed_persons: number;
    active_critical_alerts: number;
    total_alerts: number;
    model_mAP50: number;
    avg_inference_ms: number;
  };
  trend_data: Array<{
    time: string;
    persons: number;
    vehicles: number;
    alerts: number;
  }>;
  class_distribution: Array<{
    name: string;
    value: number;
    color: string;
  }>;
  priority_counts: Array<{
    priority: string;
    count: number;
    color: string;
  }>;
  recent_alerts: Alert[];
}

export interface DatasetStats {
  dataset_name: string;
  classes: Record<string, string>;
  splits: {
    train: { image_count?: number; images?: number; box_count?: number; boxes?: number };
    val: { image_count?: number; images?: number; box_count?: number; boxes?: number };
    test: { image_count?: number; images?: number; box_count?: number; boxes?: number };
  };
  class_counts: Record<string, number>;
  source: string;
  license: string;
}

export interface DatasetSample {
  id: string;
  url: string;
  filename: string;
  title: string;
}

export interface PerClassMetric {
  class: string;
  precision: number;
  recall: number;
  mAP50: number;
  mAP50_95: number;
}

export interface ModelMetrics {
  architecture: string;
  parameters: string;
  input_resolution: string;
  overall?: {
    precision: number;
    recall: number;
    mAP50: number;
    mAP50_95: number;
  };
  mAP50: number;
  mAP50_95: number;
  precision: number;
  recall: number;
  per_class: PerClassMetric[];
  training_epochs: number;
  batch_size: number;
  training_split?: string;
}

export interface ModelAssets {
  results_png: string;
  confusion_matrix: string;
  pr_curve: string;
  f1_curve: string;
  weights_url: string;
  weights_size_mb: number;
}

export interface AnalysisJob {
  job_id: string;
  filename: string;
  status: 'queued' | 'processing' | 'completed' | 'error';
  progress: number;
  created_at: string;
  is_video: boolean;
  error?: string;
  result?: {
    media_type: 'image' | 'video';
    result_url: string;
    total_entities: number;
    frames_processed?: number;
    boxes?: BoxDetection[];
    summary: {
      persons: number;
      armed_persons: number;
      vehicles: number;
      military_vehicles: number;
      weapons: number;
    };
  };
}

