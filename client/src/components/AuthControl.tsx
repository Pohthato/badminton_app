import React from "react";
import { MoreHorizontal } from "lucide-react";
import type { AuthAction } from "@/lib/authUi";

export function AuthControl({
  action,
  label,
  profileMenuOpen,
  onAuthClick,
  onLogout,
}: {
  action: AuthAction;
  label: string;
  profileMenuOpen: boolean;
  onAuthClick: () => void;
  onLogout: () => void;
}) {
  const isAuthenticated = action === "profile";
  return (
    <div className="flex items-center gap-3">
      <div className="hidden items-center gap-2 rounded-full border border-[#b7fa59]/20 bg-[#b7fa59]/5 px-3 py-1.5 text-[11px] font-semibold text-[#ceff90] sm:flex"><span aria-hidden="true">◈</span> Private session</div>
      <button className="rounded-full border border-white/10 bg-white/5 p-2 text-slate-300 transition-colors hover:bg-white/10" aria-label="More session actions"><MoreHorizontal className="h-5 w-5" /></button>
      <div className="relative">
        <button onClick={onAuthClick} aria-label={label} title={label} className="flex h-9 w-9 items-center justify-center rounded-full border border-[#b7fa59]/30 bg-gradient-to-br from-[#4a6852] to-[#1c3131] text-xs font-bold text-[#dfffb6] transition-transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-[#b7fa59]/60">{isAuthenticated ? "S" : "?"}</button>
        {profileMenuOpen && isAuthenticated && <div className="absolute right-0 top-11 z-30 w-44 rounded-xl border border-white/10 bg-[#101a1f] p-2 shadow-2xl"><p className="px-2 py-1.5 text-[10px] uppercase tracking-[.14em] text-slate-500">Account</p><button onClick={onLogout} className="w-full rounded-lg px-2 py-2 text-left text-xs font-semibold text-slate-300 hover:bg-white/5 hover:text-white">Sign out</button></div>}
      </div>
    </div>
  );
}
