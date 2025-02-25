# Generated by Django 4.2.5 on 2023-09-29 10:14

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import utrack.db.models.workspace
import uuid


def update_issue_activity_priority(apps, schema_editor):
    IssueActivity = apps.get_model("db", "IssueActivity")
    updated_issue_activity = []
    for obj in IssueActivity.objects.filter(field="priority"):
        # Set the old and new value to none if it is empty for Priority
        obj.new_value = obj.new_value or "none"
        obj.old_value = obj.old_value or "none"
        updated_issue_activity.append(obj)
    IssueActivity.objects.bulk_update(
        updated_issue_activity,
        ["new_value", "old_value"],
        batch_size=2000,
    )


def update_issue_activity_blocked(apps, schema_editor):
    IssueActivity = apps.get_model("db", "IssueActivity")
    updated_issue_activity = []
    for obj in IssueActivity.objects.filter(field="blocks"):
        # Set the field to blocked_by
        obj.field = "blocked_by"
        updated_issue_activity.append(obj)
    IssueActivity.objects.bulk_update(
        updated_issue_activity,
        ["field"],
        batch_size=1000,
    )


class Migration(migrations.Migration):
    dependencies = [
        ("db", "0044_auto_20230913_0709"),
    ]

    operations = [
        migrations.CreateModel(
            name="GlobalView",
            fields=[
                (
                    "created_at",
                    models.DateTimeField(
                        auto_now_add=True, verbose_name="Created At"
                    ),
                ),
                (
                    "updated_at",
                    models.DateTimeField(
                        auto_now=True, verbose_name="Last Modified At"
                    ),
                ),
                (
                    "id",
                    models.UUIDField(
                        db_index=True,
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                        unique=True,
                    ),
                ),
                (
                    "name",
                    models.CharField(max_length=255, verbose_name="View Name"),
                ),
                (
                    "description",
                    models.TextField(
                        blank=True, verbose_name="View Description"
                    ),
                ),
                ("query", models.JSONField(verbose_name="View Query")),
                (
                    "access",
                    models.PositiveSmallIntegerField(
                        choices=[(0, "Private"), (1, "Public")], default=1
                    ),
                ),
                ("query_data", models.JSONField(default=dict)),
                ("sort_order", models.FloatField(default=65535)),
                (
                    "created_by",
                    models.ForeignKey(
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="%(class)s_created_by",
                        to=settings.AUTH_USER_MODEL,
                        verbose_name="Created By",
                    ),
                ),
                (
                    "updated_by",
                    models.ForeignKey(
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="%(class)s_updated_by",
                        to=settings.AUTH_USER_MODEL,
                        verbose_name="Last Modified By",
                    ),
                ),
                (
                    "workspace",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="global_views",
                        to="db.workspace",
                    ),
                ),
            ],
            options={
                "verbose_name": "Global View",
                "verbose_name_plural": "Global Views",
                "db_table": "global_views",
                "ordering": ("-created_at",),
            },
        ),
        migrations.AddField(
            model_name="workspacemember",
            name="issue_props",
            field=models.JSONField(
                default=utrack.db.models.workspace.get_issue_props
            ),
        ),
        migrations.AddField(
            model_name="issueactivity",
            name="epoch",
            field=models.FloatField(null=True),
        ),
        migrations.RunPython(update_issue_activity_priority),
        migrations.RunPython(update_issue_activity_blocked),
    ]
