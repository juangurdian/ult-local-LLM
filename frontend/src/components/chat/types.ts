export type Role = "user" | "assistant" | "system";

export type Message = {
  id: string;
  role: Role;
  content: string;
  createdAt: number;
};

export type Conversation = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: Message[];
};

export type RoutingInfo = {
  model: string;
  routing_method: string;
  task_type?: string;
  confidence?: number;
  complexity?: string;
  reasoning?: string;
  estimated_tokens?: number;
  packing?: {
    context_window?: number;
    target_tokens?: number;
    tokens_total?: number;
    tokens_kept?: number;
    tokens_dropped?: number;
    used_summary?: boolean;
    summary_tokens?: number;
  };
  processing_time_ms?: number;
  timestamp?: string;
  warning?: string;
  model_meta?: {
    context_window?: number;
    estimated_vram_gb?: number;
    estimated_tokens_per_sec?: number;
  };
  classification_details?: {
    task_type?: string;
    confidence?: number;
    complexity_score?: number;
    reasoning?: string;
  };
};

