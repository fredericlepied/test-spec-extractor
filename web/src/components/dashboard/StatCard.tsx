import { useNavigate } from "react-router-dom";

interface Props {
  label: string;
  value: number | string;
  detail?: string;
  to?: string;
}

export function StatCard({ label, value, detail, to }: Props) {
  const navigate = useNavigate();
  return (
    <div
      onClick={to ? () => navigate(to) : undefined}
      className={`bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 ${to ? "cursor-pointer hover:border-blue-400 dark:hover:border-blue-500 transition-colors" : ""}`}
    >
      <div className="text-sm text-gray-500 dark:text-gray-400">{label}</div>
      <div className="text-3xl font-bold mt-1">{typeof value === "number" ? value.toLocaleString() : value}</div>
      {detail && <div className="text-xs text-gray-400 dark:text-gray-500 mt-1">{detail}</div>}
    </div>
  );
}
