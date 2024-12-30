// assets
import packageJson from "package.json";

export const UtrackVersionNumber: React.FC = () => <span>Version: v{packageJson.version}</span>;
