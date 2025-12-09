"use client";
import { FormEvent, KeyboardEvent, useState, useRef, useEffect } from "react";

type MessageInputProps = {
  disabled?: boolean;
  onSend: (content: string) => void;
};

export default function MessageInput({ disabled, onSend }: MessageInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea based on content
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`;
    }
  }, [value]);

  const handleSubmit = (e?: FormEvent) => {
    e?.preventDefault();
    const content = value.trim();
    if (!content || disabled) return;
    onSend(content);
    setValue("");
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter without Shift sends the message
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
    // Shift+Enter allows newline (default behavior)
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask anything… (Enter to send, Shift+Enter for newline)"
        rows={1}
        disabled={disabled}
        className="max-h-[150px] min-h-[44px] w-full resize-none rounded-xl border border-slate-800 bg-slate-950/70 px-4 py-3 text-sm text-slate-100 shadow-inner shadow-slate-950/30 outline-none ring-1 ring-transparent transition focus:ring-cyan-500 disabled:cursor-not-allowed disabled:opacity-60"
      />
      <div className="flex items-center justify-between">
        <div className="text-xs text-slate-500">
          <kbd className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px]">Enter</kbd>
          {" "}to send
          <span className="mx-2 text-slate-700">•</span>
          <kbd className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px]">Shift</kbd>
          {" + "}
          <kbd className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px]">Enter</kbd>
          {" "}for newline
        </div>
        <button
          type="submit"
          disabled={disabled || !value.trim()}
          className="inline-flex items-center gap-2 rounded-lg bg-cyan-500 px-4 py-2 text-sm font-semibold text-slate-900 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
          </svg>
          Send
        </button>
      </div>
    </form>
  );
}
