interface Props {
  score: number;
}

export function ScoreBadge({ score }: Props) {
  const pct = (score * 100).toFixed(1);
  let colors: string;
  if (score >= 0.95) colors = "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-300";
  else if (score >= 0.85) colors = "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-300";
  else if (score >= 0.75) colors = "bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-300";
  else colors = "bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300";

  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${colors}`}>
      {pct}%
    </span>
  );
}
