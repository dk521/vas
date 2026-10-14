from django.core.management.base import BaseCommand
from apps.voice.models.pangrams import Pangram

PANGRAMS = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump.",
    "The five boxing wizards jump quickly.",
    "Sphinx of black quartz, judge my vow.",
    "Jackdaws love my big sphinx of quartz.",
    "Quick zephyrs blow, vexing daft Jim.",
    "Two driven jocks help fax my big quiz.",
    "Five quacking zephyrs jolt my wax bed.",
    "The jay, pig, fox, zebra, and my wolves quack.",
]


class Command(BaseCommand):
    help = 'Seed pangrams for voice enrollment'

    def handle(self, *args, **options):
        created_count = 0

        for text in PANGRAMS:
            _, created = Pangram.objects.get_or_create(
                text = text,
                defaults={'is_active': True}
            )

            if created:
                created_count += 1
        self.stdout.write(
            self.style.SUCCESS(
                f'Seed complete. Created {created_count} pangrams in Database'
            )
        )