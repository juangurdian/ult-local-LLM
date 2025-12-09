"use client";
"use client";
import { useMemo } from "react";
import { useAppStore, AppMode } from "@/lib/stores/app";
import { useOffline } from "@/lib/hooks/useOffline";

const MODELS = [
  { id: "qwen3-4b", label: "qwen3:4b", desc: "Fast chat" },
  { id: "qwen3-8b", label: "qwen3:8b", desc: "Balanced" },
  { id: "deepseek-r1-8b", label: "deepseek-r1:8b", desc: "Reasoning" },
  { id: "qwen2.5-coder-7b", label: "qwen2.5-coder:7b", desc: "Coding" },
];

export default function Header() {
  const mode = useAppStore((s) => s.mode);
  const setMode = useAppStore((s) => s.setMode);
  const offline = useOffline();
  const status = useMemo(
    () => [
      { label: "Router", state: offline ? "offline" : "online", color: offline ? "bg-amber-400" : "bg-emerald-400" },
      { label: "Ollama", state: offline ? "offline" : "online", color: offline ? "bg-amber-400" : "bg-emerald-400" },
    ],
    [offline]
  );

  return (
    <header className="shrink-0 border-b border-slate-900/70 bg-slate-950/90 px-4 py-3 backdrop-blur sm:px-6 lg:px-8">
      <div className="flex items-center justify-between gap-4">
        {/* Left: Title + Status */}
        <div className="flex items-center gap-4">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold text-slate-50">Local AI Beast</h1>
              <span className="rounded-full bg-cyan-500/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-cyan-300">
                {mode.charAt(0).toUpperCase() + mode.slice(1)}
              </span>
            </div>
            <p className="text-xs text-slate-500">
              {offline ? "Offline mode: local-only" : "Offline-first • Router-aware"}
            </p>
          </div>
          
          {/* Status indicators */}
          <div className="hidden items-center gap-2 sm:flex">
            {status.map((item) => (
              <span
                key={item.label}
                className="inline-flex items-center gap-1.5 rounded-full bg-slate-900/80 px-2.5 py-1 text-[10px] text-slate-300 ring-1 ring-slate-800"
              >
                <span className={`h-1.5 w-1.5 rounded-full ${item.color}`} />
                {item.label}
              </span>
            ))}
          </div>
        </div>

        {/* Right: Model selector */}
        <div className="flex items-center gap-2">
          <label className="hidden text-[10px] uppercase tracking-wider text-slate-500 sm:block">
            Model
          </label>
          <select
            className="min-w-[160px] rounded-lg border border-slate-800 bg-slate-900 px-3 py-1.5 text-xs text-slate-100 outline-none ring-1 ring-transparent transition focus:ring-cyan-500"
            defaultValue="qwen3-8b"
          >
            {MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label} — {m.desc}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Agent mode tabs */}
      <div className="mt-3 flex flex-wrap gap-2">
        {(["chat", "research", "code", "image"] as AppMode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`rounded-lg border px-3 py-1.5 text-xs font-semibold transition ${
              mode === m
                ? "border-cyan-500 bg-cyan-500/20 text-cyan-100"
                : "border-slate-800 bg-slate-900/60 text-slate-200 hover:border-cyan-500/50 hover:text-cyan-200"
            }`}
          >
            {m === "chat" ? "Chat" : m === "research" ? "Research" : m === "code" ? "Code" : "Image"}
          </button>
        ))}
      </div>
    </header>
  );
}
