import React, {useMemo} from 'react';
import Link from 'next/link';

// Renders a text with catalog cohort names turned into purple underlined links
// that open the explore page with that cohort's section expanded
// (/cohorts?cohort=<id>). Matching is case-insensitive and tolerant of
// hyphen/space/unicode-dash variance, but the DISPLAYED text is exactly what
// the writer typed - only the link target uses the canonical id.

const SEP_SRC = '[\\s\\u2010-\\u2015\\u2212-]+';

export default function CohortLinkedText({
  text,
  names,
  className
}: {
  text: string;
  names: string[];
  className?: string;
}) {
  const namesKey = (names || []).join('\n');
  const nodes = useMemo(() => {
    const canon = Array.from(new Set((names || []).filter(n => n && n.length >= 3)));
    if (!text || canon.length === 0) return [text];
    const escRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const flexible = (n: string) => n.split(new RegExp(SEP_SRC)).map(escRe).join(SEP_SRC);
    const norm = (s: string) => s.toLowerCase().replace(new RegExp(SEP_SRC, 'g'), ' ');
    const byNorm = new Map(canon.map(n => [norm(n), n]));
    const re = new RegExp(
      `(^|[^\\w-])(${canon.sort((a, b) => b.length - a.length).map(flexible).join('|')})(?![\\w-])`,
      'gi'
    );
    const out: React.ReactNode[] = [];
    let last = 0;
    let k = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
      const hit = m[2];
      const start = m.index + m[1].length;
      const id = byNorm.get(norm(hit));
      if (!id) continue;
      if (start > last) out.push(text.slice(last, start));
      out.push(
        <Link
          key={k++}
          href={{pathname: '/cohorts', query: {cohort: id}}}
          className="text-purple-700 dark:text-purple-400 font-semibold underline underline-offset-2 hover:text-purple-900"
          title={`Open ${id} on the explore page`}
        >
          {hit}
        </Link>
      );
      last = start + hit.length;
    }
    if (last < text.length) out.push(text.slice(last));
    return out;
    // names identity changes every render upstream; key on the joined value
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text, namesKey]);
  return <span className={className}>{nodes}</span>;
}
