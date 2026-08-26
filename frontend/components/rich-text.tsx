"use client";
import React from "react";

/**
 * Lightweight markdown renderer tuned for LLM explanations.
 * Supports: ##/# headings, -/* bullets, 1. ordered lists,
 * **bold**, *italic*, `code`, blank-line paragraphs.
 * No dependencies — safe subset, never dangerouslySetInnerHTML.
 */

function inline(text: string, keyBase: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  // tokenize: code → bold → italic
  const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*\n]+\*)/g;
  let last = 0, m: RegExpExecArray | null, k = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("`")) {
      nodes.push(<code key={`${keyBase}-c${k++}`} className="px-1.5 py-0.5 mx-0.5 rounded-md bg-sand border border-line font-mono text-[0.85em] text-ink">{tok.slice(1, -1)}</code>);
    } else if (tok.startsWith("**")) {
      nodes.push(<strong key={`${keyBase}-b${k++}`} className="font-semibold text-ink">{tok.slice(2, -2)}</strong>);
    } else {
      nodes.push(<em key={`${keyBase}-i${k++}`} className="italic">{tok.slice(1, -1)}</em>);
    }
    last = m.index + tok.length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

export default function RichText({ text, className = "" }: { text: string; className?: string }) {
  const lines = text.replace(/\r/g, "").split("\n");
  const blocks: React.ReactNode[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  let para: string[] = [];
  let key = 0;

  const flushList = () => {
    if (!list) return;
    const Tag = list.ordered ? "ol" : "ul";
    blocks.push(
      <Tag key={`l${key++}`} className={
        list.ordered
          ? "my-2 space-y-1.5 list-decimal pl-5 marker:text-accent marker:font-semibold"
          : "my-2 space-y-1.5 pl-1"
      }>
        {list.items.map((it, i) => (
          <li key={i} className={list.ordered ? "pl-1" : "relative pl-4 before:absolute before:left-0 before:top-[0.55em] before:w-1.5 before:h-1.5 before:rounded-full before:bg-accent/70"}>
            {inline(it, `li${key}-${i}`)}
          </li>
        ))}
      </Tag>
    );
    list = null;
  };
  const flushPara = () => {
    if (!para.length) return;
    blocks.push(
      <p key={`p${key++}`} className="my-2 leading-relaxed first:mt-0 last:mb-0">
        {inline(para.join(" "), `pp${key}`)}
      </p>
    );
    para = [];
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) { flushList(); flushPara(); continue; }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    const ul = line.match(/^\s*[-*•]\s+(.*)$/);
    const ol = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
    if (h) { flushList(); flushPara();
      blocks.push(
        <div key={`h${key++}`} className="mt-3 mb-1 flex items-center gap-2">
          <span className="w-4 h-[2px] rounded bg-accent/60" />
          <span className="text-sm font-semibold tracking-wide text-ink">{inline(h[2], `hh${key}`)}</span>
        </div>
      );
      continue;
    }
    if (ul) { flushPara(); if (!list || list.ordered) { flushList(); list = { ordered: false, items: [] }; }
      list.items.push(ul[1]); continue; }
    if (ol) { flushPara(); if (!list || !list.ordered) { flushList(); list = { ordered: true, items: [] }; }
      list.items.push(ol[2]); continue; }
    flushList(); para.push(line.trim());
  }
  flushList(); flushPara();

  return <div className={`text-sm text-muted ${className}`}>{blocks}</div>;
}
