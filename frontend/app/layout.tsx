import "./globals.css";
import { Inter, Newsreader } from "next/font/google";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });
const news = Newsreader({ subsets: ["latin"], variable: "--font-serif" });

export const metadata = {
  title: "FusionUncertaintyNet — Protein Reliability",
  description: "Adaptive Multi-PLM with Evidential Deep Learning for calibrated protein structure reliability",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${news.variable}`}>
      <body className="font-sans antialiased min-h-screen bg-paper">
        <header className="sticky top-0 z-20 backdrop-blur bg-paper/80 border-b border-line">
          <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center text-white font-serif text-lg">◈</div>
              <div>
                <div className="font-serif text-[18px] leading-none tracking-tight">FusionUncertaintyNet</div>
                <div className="text-xs text-muted -mt-0.5">Calibrated Protein Reliability</div>
              </div>
            </div>
            <nav className="flex items-center gap-6 text-sm">
              <a href="/" className="hover:text-accent">Overview</a>
              <a href="/dashboard" className="hover:text-accent">Predict</a>
              <a href="/history" className="hover:text-accent">History</a>
              <a href="/login" className="px-4 py-2 rounded-full bg-ink text-white hover:bg-black transition">Sign in</a>
            </nav>
          </div>
        </header>
        <main className="max-w-6xl mx-auto px-6 py-8">{children}</main>
        <footer className="max-w-6xl mx-auto px-6 py-12 text-xs text-muted border-t border-line mt-16">
          <div>Built for structural biology · ESM-2 + ProtT5 + AF features · Gamma EDR · Firebase cabbage-guard (fusion_*)</div>
          <div className="mt-1">Heavy inference on Hugging Face Spaces (bhumika-tewari-282006) · Lite on Render · Frontend on Vercel</div>
        </footer>
      </body>
    </html>
  );
}
