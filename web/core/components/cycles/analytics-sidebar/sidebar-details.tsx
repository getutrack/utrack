"use client";
import React, { FC } from "react";
import isEmpty from "lodash/isEmpty";
import { observer } from "mobx-react";
import { LayersIcon, SquareUser, Users } from "lucide-react";
// utrack types
import { ICycle } from "@utrack/types";
// utrack ui
import { Avatar, AvatarGroup, TextArea } from "@utrack/ui";
// helpers
import { getFileURL } from "@/helpers/file.helper";
// hooks
import { useMember, useProjectEstimates } from "@/hooks/store";
// utrack web constants
import { EEstimateSystem } from "@/utrack-web/constants/estimates";

type Props = {
  projectId: string;
  cycleDetails: ICycle;
};

export const CycleSidebarDetails: FC<Props> = observer((props) => {
  const { projectId, cycleDetails } = props;
  // hooks
  const { getUserDetails } = useMember();
  const { areEstimateEnabledByProjectId, currentActiveEstimateId, estimateById } = useProjectEstimates();

  const areEstimateEnabled = projectId && areEstimateEnabledByProjectId(projectId.toString());
  const cycleStatus = cycleDetails?.status?.toLocaleLowerCase();
  const isCompleted = cycleStatus === "completed";

  const issueCount =
    isCompleted && !isEmpty(cycleDetails?.progress_snapshot)
      ? cycleDetails?.progress_snapshot?.total_issues === 0
        ? "0 Issue"
        : `${cycleDetails?.progress_snapshot?.completed_issues}/${cycleDetails?.progress_snapshot?.total_issues}`
      : cycleDetails?.total_issues === 0
        ? "0 Issue"
        : `${cycleDetails?.completed_issues}/${cycleDetails?.total_issues}`;
  const estimateType = areEstimateEnabled && currentActiveEstimateId && estimateById(currentActiveEstimateId);
  const cycleOwnerDetails = cycleDetails ? getUserDetails(cycleDetails.owned_by_id) : undefined;

  const isEstimatePointValid = isEmpty(cycleDetails?.progress_snapshot || {})
    ? estimateType && estimateType?.type == EEstimateSystem.POINTS
      ? true
      : false
    : isEmpty(cycleDetails?.progress_snapshot?.estimate_distribution || {})
      ? false
      : true;

  const issueEstimatePointCount =
    isCompleted && !isEmpty(cycleDetails?.progress_snapshot)
      ? cycleDetails?.progress_snapshot.total_issues === 0
        ? "0 Issue"
        : `${cycleDetails?.progress_snapshot.completed_estimate_points}/${cycleDetails?.progress_snapshot.total_estimate_points}`
      : cycleDetails?.total_issues === 0
        ? "0 Issue"
        : `${cycleDetails?.completed_estimate_points}/${cycleDetails?.total_estimate_points}`;
  return (
    <div className="flex flex-col gap-5 w-full">
      {cycleDetails?.description && (
        <TextArea
          className="outline-none ring-none w-full max-h-max bg-transparent !p-0 !m-0 !border-0 resize-none text-sm leading-5 text-custom-text-200"
          value={cycleDetails.description}
          disabled
        />
      )}

      <div className="flex flex-col gap-5 pb-6 pt-2.5">
        <div className="flex items-center justify-start gap-1">
          <div className="flex w-2/5 items-center justify-start gap-2 text-custom-text-300">
            <SquareUser className="h-4 w-4" />
            <span className="text-base">Lead</span>
          </div>
          <div className="flex w-3/5 items-center rounded-sm">
            <div className="flex items-center gap-2.5">
              <Avatar name={cycleOwnerDetails?.display_name} src={getFileURL(cycleOwnerDetails?.avatar_url ?? "")} />
              <span className="text-sm text-custom-text-200">{cycleOwnerDetails?.display_name}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-start gap-1">
          <div className="flex w-2/5 items-center justify-start gap-2 text-custom-text-300">
            <Users className="h-4 w-4" />
            <span className="text-base">Members</span>
          </div>
          <div className="flex w-3/5 items-center rounded-sm">
            <div className="flex items-center gap-2.5">
              {cycleDetails?.assignee_ids && cycleDetails.assignee_ids.length > 0 ? (
                <>
                  <AvatarGroup showTooltip>
                    {cycleDetails.assignee_ids.map((member) => {
                      const memberDetails = getUserDetails(member);
                      return (
                        <Avatar
                          key={memberDetails?.id}
                          name={memberDetails?.display_name ?? ""}
                          src={getFileURL(memberDetails?.avatar_url ?? "")}
                          showTooltip={false}
                        />
                      );
                    })}
                  </AvatarGroup>
                </>
              ) : (
                <span className="px-1.5 text-sm text-custom-text-300">No assignees</span>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center justify-start gap-1">
          <div className="flex w-2/5 items-center justify-start gap-2 text-custom-text-300">
            <LayersIcon className="h-4 w-4" />
            <span className="text-base">Issues</span>
          </div>
          <div className="flex w-3/5 items-center">
            <span className="px-1.5 text-sm text-custom-text-300">{issueCount}</span>
          </div>
        </div>

        {/**
         * NOTE: Render this section when estimate points of he projects is enabled and the estimate system is points
         */}
        {isEstimatePointValid && (
          <div className="flex items-center justify-start gap-1">
            <div className="flex w-2/5 items-center justify-start gap-2 text-custom-text-300">
              <LayersIcon className="h-4 w-4" />
              <span className="text-base">Points</span>
            </div>
            <div className="flex w-3/5 items-center">
              <span className="px-1.5 text-sm text-custom-text-300">{issueEstimatePointCount}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
});
