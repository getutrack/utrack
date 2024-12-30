// editor
import { TEmbedConfig } from "@utrack/editor";
// types
import { TPageEmbedType } from "@utrack/types";
// utrack web components
import { IssueEmbedUpgradeCard } from "@/utrack-web/components/pages";

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
