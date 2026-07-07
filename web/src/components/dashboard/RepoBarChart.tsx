import { useNavigate } from "react-router-dom";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

const COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"];

interface Props {
  data: { name: string; count: number }[];
}

export function RepoBarChart({ data }: Props) {
  const navigate = useNavigate();

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
      <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-4">Tests by Repository</h3>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart
          data={data}
          margin={{ bottom: 60 }}
          onClick={(state) => {
            if (state?.activeLabel) navigate(`/catalog?repo=${state.activeLabel}`);
          }}
          style={{ cursor: "pointer" }}
        >
          <XAxis dataKey="name" angle={-35} textAnchor="end" fontSize={11} tick={{ fill: "#9ca3af" }} />
          <YAxis fontSize={11} tick={{ fill: "#9ca3af" }} />
          <Tooltip
            contentStyle={{ backgroundColor: "var(--color-tooltip-bg)", border: "1px solid var(--color-tooltip-border)", borderRadius: 8, color: "var(--color-tooltip-text)" }}
            labelStyle={{ color: "var(--color-tooltip-text)" }}
            itemStyle={{ color: "var(--color-tooltip-text)" }}
          />
          <Bar dataKey="count" radius={[4, 4, 0, 0]}>
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
