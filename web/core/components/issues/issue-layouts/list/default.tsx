import { useEffect, useRef } from "react";
import { combine } from "@atlaskit/pragmatic-drag-and-drop/combine";
import { autoScrollForElements } from "@atlaskit/pragmatic-drag-and-drop-auto-scroll/element";
import { observer } from "mobx-react";
// utrack constants
import { ALL_ISSUES } from "@utrack/constants";
// types
import {
  GroupByColumnTypes,
  TGroupedIssues,
  TIssue,
  IIssueDisplayProperties,
  TIssueMap,
  TIssueGroupByOptions,
  TIssueOrderByOptions,
  IGroupByColumn,
  TIssueKanbanFilters,
} from "@utrack/types";
// components
import { MultipleSelectGroup } from "@/components/core";

// hooks
import { useCycle, useLabel, useMember, useModule, useProject, useProjectState } from "@/hooks/store";
import { useIssueStoreType } from "@/hooks/use-issue-layout-store";
// utrack web components
import { IssueBulkOperationsRoot } from "@/utrack-web/components/issues";
// utrack web hooks
import { useBulkOperationStatus } from "@/utrack-web/hooks/use-bulk-operation-status";
// utils
import { getGroupByColumns, isWorkspaceLevel, GroupDropLocation, isSubGrouped } from "../utils";
import { ListGroup } from "./list-group";
import { TRenderQuickActions } from "./list-view-types";

export interface IList {
  groupedIssueIds: TGroupedIssues;
  issuesMap: TIssueMap;
  group_by: TIssueGroupByOptions | null;
  orderBy: TIssueOrderByOptions | undefined;
  updateIssue: ((projectId: string | null, issueId: string, data: Partial<TIssue>) => Promise<void>) | undefined;
  quickActions: TRenderQuickActions;
  displayProperties: IIssueDisplayProperties | undefined;
  enableIssueQuickAdd: boolean;
  showEmptyGroup?: boolean;
  canEditProperties: (projectId: string | undefined) => boolean;
  quickAddCallback?: (projectId: string | null | undefined, data: TIssue) => Promise<TIssue | undefined>;
  disableIssueCreation?: boolean;
  handleOnDrop: (source: GroupDropLocation, destination: GroupDropLocation) => Promise<void>;
  addIssuesToView?: (issueIds: string[]) => Promise<TIssue>;
  isCompletedCycle?: boolean;
  loadMoreIssues: (groupId?: string) => void;
  handleCollapsedGroups: (value: string) => void;
  collapsedGroups: TIssueKanbanFilters;
}

export const List: React.FC<IList> = observer((props) => {
  const {
    groupedIssueIds,
    issuesMap,
    group_by,
    orderBy,
    updateIssue,
    quickActions,
    displayProperties,
    enableIssueQuickAdd,
    showEmptyGroup,
    canEditProperties,
    quickAddCallback,
    disableIssueCreation,
    handleOnDrop,
    addIssuesToView,
    isCompletedCycle = false,
    loadMoreIssues,
    handleCollapsedGroups,
    collapsedGroups,
  } = props;

  const storeType = useIssueStoreType();
  // store hooks
  const member = useMember();
  const project = useProject();
  const label = useLabel();
  const projectState = useProjectState();
  const cycle = useCycle();
  const projectModule = useModule();
  // utrack web hooks
  const isBulkOperationsEnabled = useBulkOperationStatus();

  const containerRef = useRef<HTMLDivElement | null>(null);

  const groups = getGroupByColumns(
    group_by as GroupByColumnTypes,
    project,
    cycle,
    projectModule,
    label,
    projectState,
    member,
    true,
    isWorkspaceLevel(storeType)
  );

  // Enable Auto Scroll for Main Kanban
  useEffect(() => {
    const element = containerRef.current;

    if (!element) return;

    return combine(
      autoScrollForElements({
        element,
      })
    );
  }, [containerRef]);

  if (!groups) return null;

  const getGroupIndex = (groupId: string | undefined) => groups.findIndex(({ id }) => id === groupId);

  const is_list = group_by === null ? true : false;

  // create groupIds array and entities object for bulk ops
  const groupIds = groups.map((g) => g.id);
  const orderedGroups: Record<string, string[]> = {};
  groupIds.forEach((gID) => {
    orderedGroups[gID] = [];
  });
  let entities: Record<string, string[]> = {};

  if (is_list) {
    entities = Object.assign(orderedGroups, { [groupIds[0]]: groupedIssueIds[ALL_ISSUES] ?? [] });
  } else if (!isSubGrouped(groupedIssueIds)) {
    entities = Object.assign(orderedGroups, { ...groupedIssueIds });
  } else {
    entities = orderedGroups;
  }
  return (
    <div className="relative size-full flex flex-col">
      {groups && (
        <MultipleSelectGroup containerRef={containerRef} entities={entities} disabled={!isBulkOperationsEnabled}>
          {(helpers) => (
            <>
              <div
                ref={containerRef}
                className="size-full vertical-scrollbar scrollbar-lg relative overflow-auto vertical-scrollbar-margin-top-md"
              >
                {groups.map((group: IGroupByColumn) => (
                  <ListGroup
                    key={group.id}
                    groupIssueIds={groupedIssueIds?.[group.id]}
                    issuesMap={issuesMap}
                    group_by={group_by}
                    group={group}
                    updateIssue={updateIssue}
                    quickActions={quickActions}
                    orderBy={orderBy}
                    getGroupIndex={getGroupIndex}
                    handleOnDrop={handleOnDrop}
                    displayProperties={displayProperties}
                    enableIssueQuickAdd={enableIssueQuickAdd}
                    showEmptyGroup={showEmptyGroup}
                    canEditProperties={canEditProperties}
                    quickAddCallback={quickAddCallback}
                    disableIssueCreation={disableIssueCreation}
                    addIssuesToView={addIssuesToView}
                    isCompletedCycle={isCompletedCycle}
                    loadMoreIssues={loadMoreIssues}
                    containerRef={containerRef}
                    selectionHelpers={helpers}
                    handleCollapsedGroups={handleCollapsedGroups}
                    collapsedGroups={collapsedGroups}
                  />
                ))}
              </div>

              <IssueBulkOperationsRoot selectionHelpers={helpers} />
            </>
          )}
        </MultipleSelectGroup>
      )}
    </div>
  );
});
