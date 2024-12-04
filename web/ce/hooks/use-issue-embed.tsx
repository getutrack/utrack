// editor
import { TEmbedConfig } from "@youtrack/editor";
// types
import { TPageEmbedType } from "@youtrack/types";
// youtrack web components
import { IssueEmbedUpgradeCard } from "@/youtrack-web/components/pages";

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export const useIssueEmbed = (workspaceSlug: string, projectId: string, queryType: TPageEmbedType = "issue") => {
  const widgetCallback = () => <IssueEmbedUpgradeCard />;

  const issueEmbedProps: TEmbedConfig["issue"] = {
    widgetCallback,
  };

  return {
    issueEmbedProps,
  };
};
