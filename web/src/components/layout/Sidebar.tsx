import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Dashboard", icon: "\u{1f4ca}" },
  { to: "/similarity", label: "Similarity", icon: "\u{1f50d}" },
  { to: "/clusters", label: "Clusters", icon: "\u{1f517}" },
  { to: "/graph", label: "Graph", icon: "\u{1f578}️" },
  { to: "/catalog", label: "Catalog", icon: "\u{1f4d6}" },
];

export function Sidebar() {
  return (
    <nav className="w-56 shrink-0 border-r border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 p-4 space-y-1">
      <div className="text-lg font-semibold text-gray-900 dark:text-white mb-6 px-3">
        Test Specs
      </div>
      {links.map((link) => (
        <NavLink
          key={link.to}
          to={link.to}
          end={link.to === "/"}
          className={({ isActive }) =>
            `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
              isActive
                ? "bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 font-medium"
                : "text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700"
            }`
          }
        >
          <span>{link.icon}</span>
          {link.label}
        </NavLink>
      ))}
    </nav>
  );
}
