// assets
import packageJson from "package.json";

export const YoutrackVersionNumber: React.FC = () => <span>Version: v{packageJson.version}</span>;
