"use client";

import { ReactNode } from "react";
import Header from "./Header";
import Sidebar from "./Sidebar";

type AppShellProps = {
  children: ReactNode;
};

export default function AppShell({ children }: AppShellProps) {
  return (
    <div className="flex h-full overflow-hidden bg-slate-950 text-slate-100">
      <Sidebar />
      <div className="flex flex-1 flex-col overflow-hidden border-l border-slate-900/60 bg-slate-950/80">
        <Header />
        <main className="flex-1 overflow-hidden px-4 pb-4 sm:px-6 lg:px-8">
          {children}
        </main>
      </div>
    </div>
  );
}
