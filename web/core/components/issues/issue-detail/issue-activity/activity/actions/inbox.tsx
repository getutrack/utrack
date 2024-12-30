import { FC } from "react";
import { observer } from "mobx-react";
// hooks
import { Intake } from "@utrack/ui";
import { useIssueDetail } from "@/hooks/store";
// components
import { IssueActivityBlockComponent } from "./";
// icons

type TIssueInboxActivity = { activityId: string; ends: "top" | "bottom" | undefined };

export const IssueInboxActivity: FC<TIssueInboxActivity> = observer((props) => {
  const { activityId, ends } = props;
  // hooks
  const {
    activity: { getActivityById },
  } = useIssueDetail();

  const activity = getActivityById(activityId);

  const getInboxActivityMessage = () => {
    switch (activity?.verb) {
      case "-1":
        return "declined this issue from intake.";
      case "0":
        return "snoozed this issue.";
      case "1":
        return "accepted this issue from intake.";
      case "2":
        return "declined this issue from intake by marking a duplicate issue.";
      default:
        return "updated intake issue status.";
    }
  };

  if (!activity) return <></>;
  return (
    <IssueActivityBlockComponent
      icon={<Intake className="h-4 w-4 flex-shrink-0 text-custom-text-200" />}
      activityId={activityId}
      ends={ends}
    >
      <>{getInboxActivityMessage()}</>
    </IssueActivityBlockComponent>
  );
});
