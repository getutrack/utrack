import { FC } from "react";
import { observer } from "mobx-react";
// hooks
import { useParams } from "next/navigation";
import { ProjectIssueQuickActions } from "@/components/issues";
import { useUserPermissions } from "@/hooks/store";
import { EUserPermissions, EUserPermissionsLevel } from "@/utrack-web/constants/user-permissions";

// components
// types
// constants
import { BaseListRoot } from "../base-list-root";

export const ProfileIssuesListLayout: FC = observer(() => {
  // router
  const { workspaceSlug, profileViewId } = useParams();
  // store
  const { allowPermissions } = useUserPermissions();

  const canEditPropertiesBasedOnProject = (projectId: string) =>
    allowPermissions(
      [EUserPermissions.ADMIN, EUserPermissions.MEMBER],
      EUserPermissionsLevel.PROJECT,
      workspaceSlug.toString(),
      projectId
    );

  return (
    <BaseListRoot
      QuickActions={ProjectIssueQuickActions}
      canEditPropertiesBasedOnProject={canEditPropertiesBasedOnProject}
      viewId={profileViewId?.toString()}
    />
  );
});
