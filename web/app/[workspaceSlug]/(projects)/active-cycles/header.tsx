"use client";

import { observer } from "mobx-react";
// ui
import { Breadcrumbs, ContrastIcon, Header } from "@utrack/ui";
// components
import { BreadcrumbLink } from "@/components/common";
// utrack web components
import { UpgradeBadge } from "@/utrack-web/components/workspace";

export const WorkspaceActiveCycleHeader = observer(() => (
  <Header>
    <Header.LeftItem>
      <Breadcrumbs>
        <Breadcrumbs.BreadcrumbItem
          type="text"
          link={
            <BreadcrumbLink
              label="Active cycles"
              icon={<ContrastIcon className="h-4 w-4 text-custom-text-300 rotate-180" />}
            />
          }
        />
      </Breadcrumbs>
      <UpgradeBadge size="md" />
    </Header.LeftItem>
  </Header>
));
