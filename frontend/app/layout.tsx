import "./globals.css";
import { Inter, Newsreader } from "next/font/google";
import { Viewport } from "next";
import SiteHeader from "@/components/site-header";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });
const news = Newsreader({ subsets: ["latin"], variable: "--font-serif" });

export const metadata = {
  title: "FusionUncertaintyNet — Protein Reliability",
  description: "Adaptive Multi-PLM with Evidential Deep Learning for calibrated protein structure reliability",
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
        <footer className="max-w-6xl mx-auto px-4 sm:px-6 py-10 sm:py-12 text-xs text-muted border-t border-line mt-12 sm:mt-16 pb-safe">
          <div>Built for structural biology · ESM-2 + ProtT5 + AF features · Gamma EDR · Firebase cabbage-guard (fusion_*)</div>
          <div className="mt-1">Heavy inference on Hugging Face Spaces (bhumika-tewari-282006) · Lite on Render · Frontend on Vercel</div>
        </footer>
      </body>
    </html>
  );
}
