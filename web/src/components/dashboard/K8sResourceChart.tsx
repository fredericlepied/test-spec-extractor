import { useNavigate } from "react-router-dom";
import { Treemap, ResponsiveContainer, Tooltip } from "recharts";

const COLORS = [
  "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
  "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
];

interface Props {
  data: { name: string; count: number }[];
}

interface TreemapContentProps {
  x: number;
  y: number;
  width: number;
  height: number;
  name: string;
  index: number;
}

function CustomContent({ x, y, width, height, name, index }: TreemapContentProps) {
  if (width < 40 || height < 20) return null;
  return (
    <g style={{ cursor: "pointer" }}>
      <rect x={x} y={y} width={width} height={height} fill={COLORS[index % COLORS.length]} rx={4} opacity={0.85} />
      <text x={x + width / 2} y={y + height / 2} textAnchor="middle" dominantBaseline="middle" fill="#fff" fontSize={width < 60 ? 9 : 11} fontWeight={500}>
        {name}
      </text>
    </g>
  );
}

export function K8sResourceChart({ data }: Props) {
  const navigate = useNavigate();
  const treemapData = data.slice(0, 30).map((d) => ({ name: d.name, size: d.count }));

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
      <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-4">Distinctive K8s Resources (common filtered)</h3>
      <ResponsiveContainer width="100%" height={300}>
        <Treemap
          data={treemapData}
          dataKey="size"
          nameKey="name"
          content={<CustomContent x={0} y={0} width={0} height={0} name="" index={0} />}
          onClick={(node) => {
            if (node?.name) navigate(`/catalog?resource=${node.name}`);
          }}
        >
          <Tooltip
            contentStyle={{ backgroundColor: "var(--color-tooltip-bg)", border: "1px solid var(--color-tooltip-border)", borderRadius: 8, color: "var(--color-tooltip-text)" }}
            labelStyle={{ color: "var(--color-tooltip-text)" }}
            itemStyle={{ color: "var(--color-tooltip-text)" }}
            formatter={(value) => [String(value), "Tests"]}
          />
        </Treemap>
      </ResponsiveContainer>
    </div>
  );
}
