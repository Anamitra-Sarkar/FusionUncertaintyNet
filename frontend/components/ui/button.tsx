import * as React from "react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";
export function Button({ className, variant="primary", ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & {variant?: "primary"|"ghost"|"outline"}) {
  const base = "inline-flex items-center justify-center rounded-full px-5 py-2.5 text-sm font-medium transition disabled:opacity-50";
  const variants: Record<string,string> = {
    primary: "bg-ink text-white hover:bg-black",
    ghost: "hover:bg-sand",
    outline: "border border-line bg-card hover:bg-sand"
  };
  return <button className={twMerge(clsx(base, variants[variant], className))} {...props} />;
}
