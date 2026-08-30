import "./globals.css";
import { Inter, Newsreader } from "next/font/google";
import { Viewport } from "next";
import SiteHeader from "@/components/site-header";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });
const news = Newsreader({ subsets: ["latin"], variable: "--font-serif" });

export const metadata = {
  title: "FusionUncertaintyNet — Protein Reliability",
  description: "Release-gated research framework for protein-structure reliability evaluation.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#FFFCF8",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${news.variable}`}>
      <body className="font-sans antialiased min-h-dvh bg-paper px-safe">
        <SiteHeader />
        <main className="max-w-6xl mx-auto px-4 sm:px-6 py-6 sm:py-8">{children}</main>
        <footer className="border-t border-line mt-12 sm:mt-16 pb-safe bg-card/60">
          <div className="h-[2px] w-full bg-gradient-to-r from-accent/0 via-accent/50 to-accent/0" />
          <div className="max-w-6xl mx-auto px-4 sm:px-6 py-10 grid gap-8 md:grid-cols-3">
            <div>
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-lg bg-accent flex items-center justify-center text-white font-serif">◈</div>
                <span className="font-serif tracking-tight">FusionUncertaintyNet</span>
              </div>
              <p className="text-xs text-muted mt-3 leading-relaxed max-w-xs">
                Research infrastructure that abstains until a reviewed immutable artifact and its supporting evidence are available.
              </p>
            </div>
            <div>
              <div className="text-xs font-semibold tracking-widest text-muted mb-3">PRODUCT</div>
              <ul className="space-y-2 text-sm">
                <li><a href="/" className="hover:text-accent transition-colors">Overview</a></li>
                <li><a href="/dashboard" className="hover:text-accent transition-colors">Research workspace</a></li>
                <li><a href="/history" className="hover:text-accent transition-colors">Research record</a></li>
                <li><a href="/calibration-evidence" className="hover:text-accent transition-colors">Calibration evidence</a></li>
              </ul>
            </div>
            <div className="md:text-right">
              <div className="inline-flex items-center gap-2 text-xs px-3 py-1.5 rounded-full border border-line bg-paper">
                <span className="w-1.5 h-1.5 rounded-full bg-accent2" />
                Release review pending
              </div>
              <div className="text-xs text-muted mt-4">© 2026 FusionUncertaintyNet. All rights reserved.</div>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
