import { twMerge } from "tailwind-merge"; import { clsx } from "clsx";
export function Textarea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea className={twMerge(clsx("w-full rounded-xl border border-line bg-card p-4 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent", props.className))} {...props} />;
}
export function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input className={twMerge(clsx("w-full rounded-xl border border-line bg-card px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-accent/20", props.className))} {...props} />;
}
