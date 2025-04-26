"use client";

import { FC } from "react";
import { observer } from "mobx-react";
import { IProject } from "@utrack/types";
import { ToggleSwitch, Tooltip, setPromiseToast } from "@utrack/ui";
// hooks
import { useEventTracker, useProject, useUser } from "@/hooks/store";
// utrack web components
import { UpgradeBadge } from "@/utrack-web/components/workspace";
// utrack web constants
import { PROJECT_FEATURES_LIST } from "@/utrack-web/constants/project/settings";

type Props = {
  workspaceSlug: string;
  projectId: string;
  isAdmin: boolean;
};

export const ProjectFeaturesList: FC<Props> = observer((props) => {
  const { workspaceSlug, projectId, isAdmin } = props;
  // store hooks
  const { captureEvent } = useEventTracker();
  const { data: currentUser } = useUser();
  const { getProjectById, updateProject } = useProject();
  // derived values
  const currentProjectDetails = getProjectById(projectId);

  const handleSubmit = async (featureKey: string, featureProperty: string) => {
    if (!workspaceSlug || !projectId || !currentProjectDetails) return;

    // capturing event
    captureEvent(`Toggle ${featureKey}`, {
      enabled: !currentProjectDetails?.[featureProperty as keyof IProject],
      element: "Project settings feature page",
    });

    // making the request to update the project feature
    const settingsPayload = {
      [featureProperty]: !currentProjectDetails?.[featureProperty as keyof IProject],
    };
    const updateProjectPromise = updateProject(workspaceSlug, projectId, settingsPayload);
    setPromiseToast(updateProjectPromise, {
      loading: "Updating project feature...",
      success: {
        title: "Success!",
        message: () => "Project feature updated successfully.",
      },
      error: {
        title: "Error!",
        message: () => "Something went wrong while updating project feature. Please try again.",
      },
    });
  };

  if (!currentUser) return <></>;

  return (
    <div className="space-y-6">
      {PROJECT_FEATURES_LIST.map((feature) => (
        <div key={feature.key} className="gap-x-8 gap-y-2 border-b border-custom-border-100 bg-custom-background-100 pb-2 pt-4">
          <div className="flex items-center justify-between">
            <div className="flex items-start gap-3">
              <div className="flex items-center justify-center rounded bg-custom-background-90 p-3">
                <feature.icon className="h-4 w-4" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h4 className="text-sm font-medium leading-5">{feature.title}</h4>
                </div>
                <p className="text-sm leading-5 tracking-tight text-custom-text-300">
                  {feature.description}
                </p>
              </div>
            </div>

            <ToggleSwitch
              value={Boolean(currentProjectDetails?.[feature.key as keyof IProject])}
              onChange={() => handleSubmit(feature.key, feature.key)}
              disabled={!feature.isAvailable || !isAdmin}
              size="sm"
            />
          </div>
        </div>
      ))}
    </div>
  );
});
