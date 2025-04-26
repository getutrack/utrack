import { observer } from "mobx-react";
// ui
import { Tooltip } from "@utrack/ui";
// hooks
import { usePlatformOS } from "@/hooks/use-platform-os";
// assets
import packageJson from "package.json";

export const WorkspaceEditionBadge = observer(() => {
  const { isMobile } = usePlatformOS();

  return (
    <Tooltip tooltipContent={`Version: v${packageJson.version}`} isMobile={isMobile}>
      <div
        tabIndex={-1}
        className="w-fit min-w-24 rounded-2xl px-2 py-1 text-center text-sm font-medium"
      >
        Utrack
      </div>
    </Tooltip>
  );
});
