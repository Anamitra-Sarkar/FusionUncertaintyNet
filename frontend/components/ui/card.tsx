import { clsx } from "clsx"; import { twMerge } from "tailwind-merge";
export function Card({className, ...p}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={twMerge(clsx("bg-card border border-line rounded-2xl shadow-sm", className))} {...p} />;
}
export function CardHeader({className, ...p}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={twMerge(clsx("p-6 pb-3", className))} {...p} />;
}
export function CardContent({className, ...p}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={twMerge(clsx("p-6 pt-0", className))} {...p} />;
}
