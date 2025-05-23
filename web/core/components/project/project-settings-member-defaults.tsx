"use client";

import { useEffect } from "react";
import { observer } from "mobx-react";
import { useParams } from "next/navigation";
import { Controller, useForm } from "react-hook-form";
import useSWR from "swr";
import { IProject, IUserLite, IWorkspace } from "@utrack/types";
// ui
import { Loader, TOAST_TYPE, ToggleSwitch, setToast } from "@utrack/ui";
// components
import { MemberSelect } from "@/components/project";
// constants
import { PROJECT_MEMBERS } from "@/constants/fetch-keys";
// hooks
import { useProject, useUserPermissions } from "@/hooks/store";
import { EUserPermissions, EUserPermissionsLevel } from "@/utrack-web/constants/user-permissions";

// types

const defaultValues: Partial<IProject> = {
  project_lead: null,
  default_assignee: null,
};

export const ProjectSettingsMemberDefaults: React.FC = observer(() => {
  // router
  const { workspaceSlug, projectId } = useParams();
  // store hooks
  const { allowPermissions } = useUserPermissions();

  const { currentProjectDetails, fetchProjectDetails, updateProject } = useProject();
  // derived values
  const isAdmin = allowPermissions(
    [EUserPermissions.ADMIN],
    EUserPermissionsLevel.PROJECT,
    currentProjectDetails?.workspace_detail?.slug,
    currentProjectDetails?.id
  );
  // form info
  const { reset, control } = useForm<IProject>({ defaultValues });
  // fetching user members
  useSWR(
    workspaceSlug && projectId ? PROJECT_MEMBERS(projectId.toString()) : null,
    workspaceSlug && projectId ? () => fetchProjectDetails(workspaceSlug.toString(), projectId.toString()) : null
  );

  useEffect(() => {
    if (!currentProjectDetails) return;

    reset({
      ...currentProjectDetails,
      default_assignee:
        (currentProjectDetails.default_assignee as IUserLite)?.id ?? currentProjectDetails.default_assignee,
      project_lead: (currentProjectDetails.project_lead as IUserLite)?.id ?? currentProjectDetails.project_lead,
      workspace: (currentProjectDetails.workspace as IWorkspace).id,
    });
  }, [currentProjectDetails, reset]);

  const submitChanges = async (formData: Partial<IProject>) => {
    if (!workspaceSlug || !projectId) return;

    reset({
      ...currentProjectDetails,
      default_assignee:
        (currentProjectDetails?.default_assignee as IUserLite)?.id ?? currentProjectDetails?.default_assignee,
      project_lead: (currentProjectDetails?.project_lead as IUserLite)?.id ?? currentProjectDetails?.project_lead,
      ...formData,
    });

    await updateProject(workspaceSlug.toString(), projectId.toString(), {
      default_assignee:
        formData.default_assignee === "none"
          ? null
          : (formData.default_assignee ?? currentProjectDetails?.default_assignee),
      project_lead:
        formData.project_lead === "none" ? null : (formData.project_lead ?? currentProjectDetails?.project_lead),
    })
      .then(() => {
        setToast({
          title: "Success!",
          type: TOAST_TYPE.SUCCESS,
          message: "Project updated successfully",
        });
      })
      .catch((err) => {
        console.error(err);
      });
  };

  const toggleGuestViewAllIssues = async (value: boolean) => {
    if (!workspaceSlug || !projectId) return;

    updateProject(workspaceSlug.toString(), projectId.toString(), {
      guest_view_all_features: value,
    })
      .then(() => {
        setToast({
          title: "Success!",
          type: TOAST_TYPE.SUCCESS,
          message: "Project updated successfully",
        });
      })
      .catch((err) => {
        console.error(err);
      });
  };

  return (
    <>
      <div className="flex items-center border-b border-custom-border-100 pb-3.5">
        <h3 className="text-xl font-medium">Defaults</h3>
      </div>

      <div className="flex w-full flex-col gap-2 pb-4">
        <div className="flex w-full items-center gap-4 py-4">
          <div className="flex w-1/2 flex-col gap-2">
            <h4 className="text-sm">Project lead</h4>
            <div className="">
              {currentProjectDetails ? (
                <Controller
                  control={control}
                  name="project_lead"
                  render={({ field: { value } }) => (
                    <MemberSelect
                      value={value}
                      onChange={(val: string) => {
                        submitChanges({ project_lead: val });
                      }}
                      isDisabled={!isAdmin}
                    />
                  )}
                />
              ) : (
                <Loader className="h-9 w-full">
                  <Loader.Item width="100%" height="100%" />
                </Loader>
              )}
            </div>
          </div>

          <div className="flex w-1/2 flex-col gap-2">
            <h4 className="text-sm">Default assignee</h4>
            <div className="">
              {currentProjectDetails ? (
                <Controller
                  control={control}
                  name="default_assignee"
                  render={({ field: { value } }) => (
                    <MemberSelect
                      value={value}
                      onChange={(val: string) => {
                        submitChanges({ default_assignee: val });
                      }}
                      isDisabled={!isAdmin}
                    />
                  )}
                />
              ) : (
                <Loader className="h-9 w-full">
                  <Loader.Item width="100%" height="100%" />
                </Loader>
              )}
            </div>
          </div>
        </div>
      </div>
      {currentProjectDetails && (
        <div className="relative pb-4 flex justify-between items-center gap-3">
          <div className="space-y-1">
            <h3 className="text-lg font-medium text-custom-text-100">
              Grant view access to all issues for guest users:
            </h3>
            <p className="text-sm text-custom-text-200">
              This will allow guests to have view access to all the project issues.
            </p>
          </div>
          <ToggleSwitch
            value={currentProjectDetails?.guest_view_all_features}
            onChange={() => toggleGuestViewAllIssues(!currentProjectDetails?.guest_view_all_features)}
            disabled={!isAdmin}
            size="md"
          />
        </div>
      )}
    </>
  );
});
