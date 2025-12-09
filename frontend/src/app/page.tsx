import AppShell from "@/components/layout/AppShell";
import ChatInterface from "@/components/chat/ChatInterface";

export default function Home() {
  return (
    <main className="h-screen overflow-hidden bg-slate-950 text-slate-100">
      <AppShell>
        <ChatInterface />
      </AppShell>
      </main>
  );
}
