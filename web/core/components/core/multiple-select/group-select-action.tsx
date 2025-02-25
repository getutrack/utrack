"use client";
// ui
import { Checkbox } from "@utrack/ui";
// helpers
import { cn } from "@/helpers/common.helper";
// hooks
import { TSelectionHelper } from "@/hooks/use-multiple-select";

type Props = {
  className?: string;
  disabled?: boolean;
  groupID: string;
  selectionHelpers: TSelectionHelper;
};

export const MultipleSelectGroupAction: React.FC<Props> = (props) => {
  const { className, disabled = false, groupID, selectionHelpers } = props;
  // derived values
  const groupSelectionStatus = selectionHelpers.isGroupSelected(groupID);

  if (selectionHelpers.isSelectionDisabled) return null;

  return (
    <Checkbox
      className={cn("size-3.5 !outline-none", className)}
      iconClassName="size-3"
      onClick={() => selectionHelpers.handleGroupClick(groupID)}
      checked={groupSelectionStatus === "complete"}
      indeterminate={groupSelectionStatus === "partial"}
      disabled={disabled}
    />
  );
};
